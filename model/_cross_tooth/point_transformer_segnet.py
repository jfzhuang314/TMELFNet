import torch
import torch.nn as nn
import einops
from thop import profile

from model._cross_tooth.PointTransformer.libs.pointops.functions import pointops
import os
from model._cross_tooth.cbl import CBLLoss
from model._cross_tooth.layer import STNkd
import torch.nn.functional as F

class LayerNorm1d(nn.BatchNorm1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (
            super()
            .forward(input.transpose(1, 2).contiguous())
            .transpose(1, 2)
            .contiguous()
        )

class BasicPointLayer(nn.Module):
    """
    Basic point layer consisting of a 1D convolution, batch normalization, leaky ReLU, and dropout.
    """

    def __init__(self, in_channels, out_channels, dropout=0.1, is_out=False):
        """
        Initializes the BasicPointLayer.
        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param dropout: Dropout probability. Default is 0.1.
        """
        super(BasicPointLayer, self).__init__()
        if is_out:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(dropout)
            )

    def forward(self, x):
        x = x.unsqueeze(0).transpose(2, 1)
        return self.conv(x).transpose(2, 1).squeeze(0)

        # return self.conv(x)



class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()

        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(
            nn.Linear(3, 3),
            LayerNorm1d(3),
            nn.ReLU(inplace=True),
            nn.Linear(3, out_planes),
        )
        self.linear_w = nn.Sequential(
            LayerNorm1d(mid_planes),
            nn.ReLU(inplace=True),
            nn.Linear(mid_planes, out_planes // share_planes),
            LayerNorm1d(out_planes // share_planes),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // share_planes, out_planes // share_planes),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)
        idx, _ = pointops.knnquery(self.nsample, p, p, o, o)
        x_k = pointops.queryandgroup(
            feat=x_k, xyz=p, offset=o, new_xyz=p, new_offset=o, nsample=self.nsample, use_xyz=True, idx=None
        )
        x_v = pointops.queryandgroup(
            feat=x_v,
            xyz=p,
            offset=o,
            new_xyz=p,
            new_offset=o,
            idx=idx,
            nsample=self.nsample,
            use_xyz=False,
        )
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        p_r = self.linear_p(p_r)
        r_qk = (
            x_k
            - x_q.unsqueeze(1)
            + einops.reduce(
                p_r, "n ns (i j) -> n ns j", reduction="sum", j=self.mid_planes
            )
        )
        w = self.linear_w(r_qk)  # (n, nsample, c)
        w = self.softmax(w)
        x = torch.einsum(
            "n t s i, n t i -> n s i",
            einops.rearrange(x_v + p_r, "n ns (s i) -> n ns s i", s=self.share_planes),
            w,
        )
        x = einops.rearrange(x, "n s i -> n (s i)")
        return x


class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i - 1].item()) // self.stride
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o)
            idx = pointops.furthestsampling(p, o, n_o)  # (m)
            n_p = p[idx.long(), :]  # (m, 3)
            x = pointops.queryandgroup(
                feat=x,
                xyz=p,
                offset=o,
                new_xyz=n_p,
                new_offset=n_o,
                nsample=self.nsample,
                use_xyz=True,
                idx=None
            )
            x = self.relu(
                self.bn(self.linear(x).transpose(1, 2).contiguous())
            )  # (m, c, nsample)
            x = self.pool(x).squeeze(-1)  # (m, c)
            p, o = n_p, n_o
        else:
            x = self.relu(self.bn(self.linear(x)))  # (n, c)
        return [p, x, o]


class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(
                nn.Linear(2 * in_planes, in_planes),
                nn.BatchNorm1d(in_planes),
                nn.ReLU(inplace=True),
            )
            self.linear2 = nn.Sequential(
                nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True)
            )
        else:
            self.linear1 = nn.Sequential(
                nn.Linear(out_planes, out_planes),
                nn.BatchNorm1d(out_planes),
                nn.ReLU(inplace=True),
            )
            self.linear2 = nn.Sequential(
                nn.Linear(in_planes, out_planes),
                nn.BatchNorm1d(out_planes),
                nn.ReLU(inplace=True),
            )

    def forward(self, pxo1, pxo2=None):
        if pxo2 is None:
            _, x, o = pxo1  # (n, 3), (n, c), (b)
            x_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i - 1], o[i], o[i] - o[i - 1]
                x_b = x[s_i:e_i, :]
                x_b = torch.cat(
                    (x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1
                )
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            x = self.linear1(x)
        else:
            p1, x1, o1 = pxo1
            p2, x2, o2 = pxo2
            x = self.linear1(x1) + pointops.interpolation(
                p2, p1, self.linear2(x2), o2, o1
            )
        return x


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super(Bottleneck, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer = PointTransformerLayer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer([p, x, o])))
        x = self.bn3(self.linear3(x))
        x += identity
        x = self.relu(x)
        return [p, x, o]


class PointTransformerSeg(nn.Module):
    def __init__(self,
                 block_net,
                 planes = [32, 64, 128, 256, 512],      # 特征数
                 blocks = [1, 1, 1, 1, 1],              # 重复层数
                 stride=[1, 4, 4, 4, 4],                # 下采样率
                 nsample=[8, 16, 16, 16, 16],           # 邻域数
                 share_planes=8,                        # 注意力头数
                 in_channels=6,
                 num_classes=15,
                 dropout=0.3,
                 pretrain=False,
                 add_cbl=True,                          # 是否添加CBL层
                 enable_pic_feat=False):
        super().__init__()

        # 1. 空间变换网络
        self.stnkd = STNkd(k=in_channels)

        # pic_feat
        self.enable_pic_feat = enable_pic_feat
        extra_pic_feat_channel = 4 if enable_pic_feat else 0
        if enable_pic_feat:
            self.extra_pic_feat = nn.Embedding(num_embeddings=17, embedding_dim=extra_pic_feat_channel)
            self.extra_pic_feat.weight.requires_grad = False
            nn.init.xavier_uniform_(self.extra_pic_feat.weight)

        self.pretrain = pretrain
        self.add_cbl = add_cbl

        self.planes = planes
        self.in_channels = in_channels + extra_pic_feat_channel
        self.in_planes = in_channels + extra_pic_feat_channel
        self.stride = stride
        self.nsample = nsample
        self.share_planes = share_planes
        self.dropout = dropout

        # fpn_planes, fpnhead_planes = 128, 64
        # stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]

        self.enc1 = self._make_enc(
            block_net,
            planes[0],
            blocks[0],
            share_planes,
            stride=stride[0],
            nsample=nsample[0],
        )  # N/1
        self.enc2 = self._make_enc(
            block_net,
            planes[1],
            blocks[1],
            share_planes,
            stride=stride[1],
            nsample=nsample[1],
        )  # N/4
        self.enc3 = self._make_enc(
            block_net,
            planes[2],
            blocks[2],
            share_planes,
            stride=stride[2],
            nsample=nsample[2],
        )  # N/16
        self.enc4 = self._make_enc(
            block_net,
            planes[3],
            blocks[3],
            share_planes,
            stride=stride[3],
            nsample=nsample[3],
        )  # N/64
        self.enc5 = self._make_enc(
            block_net,
            planes[4],
            blocks[4],
            share_planes,
            stride=stride[4],
            nsample=nsample[4],
        )  # N/256

        if not self.pretrain:
            self.dec5 = self._make_dec(
                block_net, planes[4], 1, share_planes, nsample=nsample[4], is_head=True
            )  # transform p5
            self.dec4 = self._make_dec(
                block_net, planes[3], 1, share_planes, nsample=nsample[3]
            )  # fusion p5 and p4
            self.dec3 = self._make_dec(
                block_net, planes[2], 1, share_planes, nsample=nsample[2]
            )  # fusion p4 and p3
            self.dec2 = self._make_dec(
                block_net, planes[1], 1, share_planes, nsample=nsample[1]
            )  # fusion p3 and p2
            self.dec1 = self._make_dec(
                block_net, planes[0], 1, share_planes, nsample=nsample[0]
            )  # fusion p2 and p1

            # dropout_rate = 0.1

            # ========== 共享骨干特征处理 ==========
            # self.shared_backbone = nn.Sequential(
            #     nn.Linear(planes[0], 64),
            #     nn.BatchNorm1d(64),
            #     nn.ReLU(inplace=True),
            #     nn.Dropout(dropout_rate)  # 👈 增加 Dropout
            # )

            # self.cls = BasicPointLayer(
            #     in_channels=planes[0],
            #     out_channels=num_classes,
            #     is_out=True
            # )
            #
            # self.edge_seg_head = BasicPointLayer(
            #     in_channels=planes[0],
            #     out_channels=2,
            #     is_out=True
            # )
            #
            # self.tte_seg_head = BasicPointLayer(
            #     in_channels=planes[0],
            #     out_channels=2,
            #     is_out=True
            # )

            self.cls = nn.Sequential(
                nn.Linear(planes[0], planes[0]),
                nn.BatchNorm1d(planes[0]),
                nn.ReLU(inplace=True),
                # nn.Dropout(self.dropout),
                nn.Linear(planes[0], num_classes),
            )

            self.edge_seg_head = nn.Sequential(
                nn.Linear(planes[0], planes[0]),
                nn.BatchNorm1d(planes[0]),
                nn.ReLU(inplace=True),
                # nn.Dropout(self.dropout),
                nn.Linear(planes[0], 2),
            )

            self.tte_seg_head = nn.Sequential(
                nn.Linear(planes[0], planes[0]),
                nn.BatchNorm1d(planes[0]),
                nn.ReLU(inplace=True),
                # nn.Dropout(self.dropout),
                nn.Linear(planes[0], 2),
            )

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = [
            TransitionDown(self.in_planes, planes * block.expansion, stride, nsample)
        ]
        self.in_planes = planes * block.expansion
        for _ in range(blocks):
            layers.append(
                block(self.in_planes, self.in_planes, share_planes, nsample=nsample)
            )
        return nn.Sequential(*layers)

    def _make_dec(
        self, block, planes, blocks, share_planes=8, nsample=16, is_head=False
    ):
        layers = [
            TransitionUp(self.in_planes, None if is_head else planes * block.expansion)
        ]
        self.in_planes = planes * block.expansion
        for _ in range(blocks):
            layers.append(
                block(self.in_planes, self.in_planes, share_planes, nsample=nsample)
            )
        return nn.Sequential(*layers)

    def forward(self, data, point_to_pixel_feat=None):
        batch_size = data.shape[0]
        point_size = data.shape[2]
        device = data.device
        data = data.permute(0, 2, 1).contiguous()   # BNC

        if self.enable_pic_feat:
            point_to_pixel_feat = point_to_pixel_feat.permute(0, 2, 1).contiguous()
            point_to_pixel_feat = self.extra_pic_feat(point_to_pixel_feat.long()).squeeze(2)
            data = torch.cat([data, point_to_pixel_feat], dim=2).contiguous()

        # feat_bnc = self.stnkd(data)
        # x0 = feat_bnc.reshape(batch_size*point_size, -1).contiguous()

        feat_bn = torch.cat([data[i] for i in range(batch_size)], dim=0).contiguous()
        coord = feat_bn[:, :3].contiguous()
        offset = torch.tensor([point_size * (i + 1) for i in range(batch_size)], device=device)


        p0 = coord
        x0 = feat_bn
        o0 = offset.int()
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])

        split5 = [o.item() for o in o5]
        for i in range(len(split5) - 1, 0, -1): split5[i] -= split5[i - 1]

        if not self.pretrain:
            x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
            x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1]
            x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
            x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
            x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]

            # 降低过拟合风险
            shared_feat = x1
            # shared_feat = F.dropout(shared_feat, p=0.3, training=self.training)

            seg_result = self.cls(shared_feat)
            split0 = [o.item() for o in o0]
            for i in range(len(split0) - 1, 0, -1): split0[i] -= split0[i - 1]
            seg_result = torch.stack(torch.split(seg_result, split0, dim=0)).permute(0, 2, 1).contiguous()

            edge_seg_result = self.edge_seg_head(shared_feat)
            edge_seg_result = torch.stack(torch.split(edge_seg_result, split0, dim=0)).permute(0, 2, 1).contiguous()

            tte_seg_result = self.tte_seg_head(shared_feat)
            tte_seg_result = torch.stack(torch.split(tte_seg_result, split0, dim=0)).permute(0, 2, 1).contiguous()

        else:
            seg_result = None
            edge_seg_result = None
            tte_seg_result = None

        if not self.add_cbl:
            return {
                 'seg_result': seg_result,
                 'edge_seg_result': edge_seg_result,
                 'tte_seg_result': tte_seg_result
                 }
        else:
            return {
                'seg_result': seg_result,
                'edge_seg_result': edge_seg_result,
                'tte_seg_result': tte_seg_result,
                'cbl_inputs': [p1, x1, o1]
            }


class PointTransformerSeg26(PointTransformerSeg):
    def __init__(self, **kwargs):
        super(PointTransformerSeg26, self).__init__(
            Bottleneck, [1, 1, 1, 1, 1], **kwargs
        )


class PointTransformerSeg38(PointTransformerSeg):
    def __init__(self, **kwargs):
        super(PointTransformerSeg38, self).__init__(
            Bottleneck, [1, 2, 2, 2, 2], **kwargs
        )


class PointTransformerSeg50(PointTransformerSeg):
    def __init__(self, **kwargs):
        super(PointTransformerSeg50, self).__init__(
            Bottleneck, [1, 2, 3, 5, 2], **kwargs
        )


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    input = torch.ones((2, 6, 16000)).cuda()
    # model = PointTransformerSeg50(in_channels=6, num_classes=15, pretrain=False, enable_pic_feat=False).cuda()
    model = PointTransformerSeg(Bottleneck,
                                blocks = [1, 1, 1, 1, 1],
                                planes=[32, 64, 128, 256, 256],  # 特征数
                                stride=[1, 2, 2, 4, 4],  # 下采样率
                                nsample=[8, 16, 16, 16, 16],  # 邻域数
                                share_planes=8,  # 注意力头数
                                in_channels=6,
                                num_classes=15,
                                pretrain=False,
                                enable_pic_feat=False).cuda()

    outputs = model(input)
    print(outputs['seg_result'].shape)
    print(outputs['edge_seg_result'].shape)
    print(outputs['tte_seg_result'].shape)
    p,x,o = outputs['cbl_inputs']
    print(p.shape) # [32000, 3]
    print(x.shape) # [32000, 32]
    print(o.shape,o) # (2,)[16000, 32000]



    flops, params = profile(model, (input,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1.0e9, params / 1.0e6))