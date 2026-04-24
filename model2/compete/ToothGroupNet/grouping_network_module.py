import torch
from model2.compete.ToothGroupNet.PointTransformer import get_model


class GroupingNetworkModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # 17类牙齿分割
        class_num = 17

        model_parameter = {
            "input_feat": 6,
            "stride": [1, 4, 4, 4, 4],
            "nsample": [36, 24, 24, 24, 24],
            "blocks": [2, 3, 4, 6, 3],
            "block_num": 5,
            "planes": [32, 64, 128, 256, 512],
            "crop_sample_size": 3072,
        }

        self.first_ins_cent_model = get_model(**model_parameter, c=model_parameter["input_feat"], k=class_num)

    def forward(self, inputs):
        """
        inputs
            inputs[0] => B, 6, 24000 : point features
            inputs[1] => B, 1, 24000 : ground truth segmentation
        """
        B, C, N = inputs.shape
        outputs = {}
        seg_result, edge_seg_result = self.first_ins_cent_model(inputs)

        return seg_result, edge_seg_result


if __name__ == '__main__':
    model = GroupingNetworkModule().cuda()
    inputs = torch.rand(2, 6, 16000).cuda()

    seg_result, edge_seg_result = model(inputs)
    print(seg_result.shape)
    print(edge_seg_result.shape)

    # -- coding: utf-8 --
    import torch
    from thop import profile

    flops, params = profile(model, (inputs,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1.0e9, params / 1.0e6))