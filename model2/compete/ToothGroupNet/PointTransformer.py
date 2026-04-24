from model2.compete.ToothGroupNet.blocks import *
from model2.compete.ToothGroupNet.config import CfgNode, load_cfg_from_cfg_file
import os

class MultiHead(nn.Module):
    def __init__(self, fdims, head_cfg, config, k):
        super().__init__()
        self.head_cfg = head_cfg
        self.ftype = get_ftype(head_cfg.ftype)[0]

        num_layers = config.num_layers
        infer_list = nn.ModuleList()
        ni_list = []

        for n, i in parse_stage(head_cfg.stage, num_layers):
            func = MLP(fdims[i], head_cfg, config, self.ftype)
            infer_list.append(func)
            ni_list.append((n, i))

        self.infer_list = infer_list
        self.ni_list = ni_list

        if head_cfg.combine.startswith('concat'):
            fdim = MLP.fkey_to_dims[head_cfg.ftype] * len(ni_list)
            self.comb_ops = torch.cat
        else:
            raise ValueError(f'not supported {head_cfg.combine}')
        # logits
        #k = config.num_classes
        if head_cfg.combine.endswith('mlp'):
            d = config.base_fdim
            self.cls = nn.Sequential(nn.Linear(fdim, d), nn.BatchNorm1d(d), nn.ReLU(inplace=True), nn.Linear(d, k))
        else:
            self.cls = nn.Linear(fdim, k)

    def upsample(self, stage_n, stage_i, stage_list):
        p, x, o = fetch_pxo(stage_n, stage_i, stage_list, self.ftype)
        if stage_i == 0:
            return x

        p0, _, o0 = fetch_pxo('up', 0, stage_list, self.ftype)
        x = pointops.interpolation(p, p0, x, o, o0, k=1)
        return x

    def forward(self, stage_list):
        collect_list = []
        for (n, i), func in zip(self.ni_list, self.infer_list):
            rst = func(stage_list[n][i], 'f_out')  # process to desired fdim
            stage_list[n][i][self.ftype] = rst  # store back
            collect_list.append(self.upsample(n, i, stage_list))  # (n, c) - potentially upsampled
        x = self.comb_ops(collect_list, 1)  # combine - NCHW
        x = self.cls(x)
        return x, stage_list


class PointTransformerSeg(nn.Module):
    def __init__(self, block, blocks, c, k, mask_head=None, planes=None, block_num=None, config=None, **kwargs):
        super().__init__()
        self.c = c
        self.in_planes = c
        self.k = k
        self.mask_head = mask_head
        self.block_num = block_num
        # fdims
        planes = config.planes

        # shared head in att
        if 'share_planes' not in config:
            config.share_planes = 8
        share_planes = config.share_planes

        fpn_planes, fpnhead_planes = 128, 64
        stride, nsample = config.stride, config.nsample
        self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes, stride=stride[0],
                                   nsample=nsample[0])  # N/1   - planes(fdims)=32,  blocks=2, nsample=8
        self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes, stride=stride[1],
                                   nsample=nsample[1])  # N/4   - planes(fdims)=64,  blocks=3, nsample=16
        if self.block_num >= 3:
            self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes, stride=stride[2],
                                       nsample=nsample[2])  # N/16  - planes(fdims)=128, blocks=4, nsample=16
            if self.block_num == 5:
                self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes, stride=stride[3],
                                           nsample=nsample[3])  # N/64  - planes(fdims)=256, blocks=6, nsample=16
                self.enc5 = self._make_enc(block, planes[4], blocks[4], share_planes, stride=stride[4],
                                           nsample=nsample[4])  # N/256 - planes(fdims)=512, blocks=3, nsample=16
                self.dec5 = self._make_dec(block, planes[4], 2, share_planes, nsample=nsample[4],
                                           is_head=bool(block_num - 1 == 4))  # transform p5
                self.dec4 = self._make_dec(block, planes[3], 2, share_planes, nsample=nsample[3],
                                           is_head=bool(block_num - 1 == 3))  # fusion p5 and p4
            self.dec3 = self._make_dec(block, planes[2], 2, share_planes, nsample=nsample[2],
                                       is_head=bool(block_num - 1 == 2))  # fusion p4 and p3
        self.dec2 = self._make_dec(block, planes[1], 2, share_planes, nsample=nsample[1],
                                   is_head=bool(block_num - 1 == 1))  # fusion p3 and p2
        self.dec1 = self._make_dec(block, planes[0], 2, share_planes, nsample=nsample[0])  # fusion p2 and p1

        self.config = config
        config.num_layers = block_num
        config.num_classes = k

        # TODO: 不进行多层级分割
        # if 'multi' in config:
        #     self.mask_head = MultiHead(planes, config.multi, config, k=2)
        #     self.cls_head = MultiHead(planes, config.multi, config, k=self.k)
        #     self.offset_head = MultiHead(planes, config.multi, config, k=3)

        self.cls_head = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]),
                                      nn.ReLU(inplace=True), nn.Linear(planes[0], k))
        self.edge_seg_head = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]),
                                      nn.ReLU(inplace=True), nn.Linear(planes[0], 2))
        self.offset_head = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]),
                                         nn.ReLU(inplace=True), nn.Linear(planes[0], 3))

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        """
        stride = 1 => TransitionDown = mlp, [block, ...]
        stride > 1 =>
        """
        layers = []
        layers.append(TransitionDown(self.in_planes, planes * block.expansion, stride, nsample))
        self.in_planes = planes * block.expansion  # expansion default to 1
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
        layers = []
        layers.append(TransitionUp(self.in_planes, None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def forward(self, inputs):

        """
        input:
            inputs[0] -> pxo -> batch_size, channel, 24000
            inputs[1] -> target -> batch_size, 24000
            inputs[2] -> pxo_prev -> batch_size, channel(32), 24000
        """
        B, C, N = inputs.shape
        pxo = inputs.permute(0, 2, 1)  # (batch_size, 24000, channel)
        x0 = pxo.reshape(-1, C)
        p0 = pxo[:, :, :3].reshape(-1, 3).contiguous()
        o0 = torch.arange(1, B + 1, dtype=torch.int).cuda()
        o0 *= N

        stage_list = {'inputs': inputs}

        if self.block_num == 5:
            p1, x1, o1 = self.enc1([p0, x0, o0])
            p2, x2, o2 = self.enc2([p1, x1, o1])
            p3, x3, o3 = self.enc3([p2, x2, o2])
            p4, x4, o4 = self.enc4([p3, x3, o3])
            p5, x5, o5 = self.enc5([p4, x4, o4])
            down_list = [
                # [p0, x0, o0],  # (n, 3), (n, in_feature_dims), (b)
                {'p_out': p1, 'f_out': x1, 'offset': o1},  # (n, 3), (n, base_fdims), (b) - default base_fdims = 32
                {'p_out': p2, 'f_out': x2, 'offset': o2},  # n_1
                {'p_out': p3, 'f_out': x3, 'offset': o3},  # n_2
                {'p_out': p4, 'f_out': x4, 'offset': o4},  # n_3
                {'p_out': p5, 'f_out': x5, 'offset': o5},  # n_4 - fdims = 512
            ]
            # for i, s in enumerate(down_list):
            #     print('\n\t'.join([str(i)] + [str(ss.shape) for ss in s]))
            stage_list['down'] = down_list

            x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[
                1]  # no upsample - concat with per-cloud mean: mlp[ x, mlp[mean(x)] ]
            x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1]
            x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
            x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
            x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]
            up_list = [
                {'p_out': p1, 'f_out': x1, 'offset': o1},  # n_0 = n, fdims = 32
                {'p_out': p2, 'f_out': x2, 'offset': o2},  # n_1
                {'p_out': p3, 'f_out': x3, 'offset': o3},  # n_2
                {'p_out': p4, 'f_out': x4, 'offset': o4},  # n_3
                {'p_out': p5, 'f_out': x5, 'offset': o5},  # n_4 - fdims = 512 (extracted through dec5 = mlps)
            ]
            stage_list['up'] = up_list

        # for i, s in enumerate(up_list):
        #     print('\n\t'.join([str(i)] + [str(ss.shape) for ss in s]))
        elif self.block_num == 3:
            p1, x1, o1 = self.enc1([p0, x0, o0])
            p2, x2, o2 = self.enc2([p1, x1, o1])
            p3, x3, o3 = self.enc3([p2, x2, o2])
            down_list = [
                # [p0, x0, o0],  # (n, 3), (n, in_feature_dims), (b)
                {'p_out': p1, 'f_out': x1, 'offset': o1},  # (n, 3), (n, base_fdims), (b) - default base_fdims = 32
                {'p_out': p2, 'f_out': x2, 'offset': o2},  # n_1
                {'p_out': p3, 'f_out': x3, 'offset': o3},  # n_2
            ]
            # for i, s in enumerate(down_list):
            #     print('\n\t'.join([str(i)] + [str(ss.shape) for ss in s]))
            stage_list['down'] = down_list

            x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3]), o3])[1]
            x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
            x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]
            up_list = [
                {'p_out': p1, 'f_out': x1, 'offset': o1},  # n_0 = n, fdims = 32
                {'p_out': p2, 'f_out': x2, 'offset': o2},  # n_1
                {'p_out': p3, 'f_out': x3, 'offset': o3},  # n_2
            ]

            stage_list['up'] = up_list

        elif self.block_num == 2:
            p1, x1, o1 = self.enc1([p0, x0, o0])
            p2, x2, o2 = self.enc2([p1, x1, o1])
            down_list = [
                {'p_out': p1, 'f_out': x1, 'offset': o1},  # (n, 3), (n, base_fdims), (b) - default base_fdims = 32
                {'p_out': p2, 'f_out': x2, 'offset': o2},  # n_1
            ]
            stage_list['down'] = down_list

            x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2]), o2])[1]
            x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]
            up_list = [
                {'p_out': p1, 'f_out': x1, 'offset': o1},  # n_0 = n, fdims = 32
                {'p_out': p2, 'f_out': x2, 'offset': o2},  # n_1
            ]

            stage_list['up'] = up_list

        # TODO: 不进行多层级分割
        cls_results = self.cls_head(x1)
        edge_seg_result = self.edge_seg_head(x1)

        cls_results = cls_results.view(B, N, self.k).permute(0, 2, 1)
        edge_seg_result = edge_seg_result.view(B, N, 2).permute(0, 2, 1)

        return cls_results, edge_seg_result


def get_model(**kwargs):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    cfg = load_cfg_from_cfg_file(os.path.join(dir_path, "default.yaml"))
    cfg = CfgNode(cfg, default='')
    cfg["input_feat"] = 6
    cfg["base_fdim"] = 32
    cfg["planes"] = [32, 64, 128, 256, 512]
    cfg["nstride"] = [1, 4, 4, 4, 4]
    cfg["nsample"] = [36, 24, 24, 24, 24]
    cfg["stride"] = [1, 4, 4, 4, 4]
    cfg["block_num"] = 5
    cfg["blocks"] = [2, 3, 4, 6, 3]
    kwargs["config"] = cfg
    model = PointTransformerSeg(PointTransformerBlock, **kwargs)
    return model


if __name__ == "__main__":
    model = get_model(c=6, k=17, blocks=[2, 3, 4, 6, 3], block_num=5).cuda()
    inputs = torch.rand(2, 6, 16000).cuda()
    # 模拟的分割标签
    targets = torch.randint(0, 16, (2, 1, 16000)).cuda()
    cls_results, edge_seg_result = model(inputs)
    print(cls_results.shape)
    print(edge_seg_result.shape)