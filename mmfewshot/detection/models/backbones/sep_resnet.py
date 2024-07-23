# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

from mmcv.cnn import build_conv_layer
from mmdet.models import ResNet
from mmdet.models.builder import BACKBONES
from torch import Tensor
from mmfewshot.detection.models.loha.loha import CombLohaModule


@BACKBONES.register_module()
class SepResNet(ResNet):
    """ResNet with `meta_conv` to handle different inputs in metarcnn and
    fsdetview.

    When input with shape (N, 3, H, W) from images, the network will use
    `conv1` as regular ResNet. When input with shape (N, 4, H, W) from (image +
    mask) the network will replace `conv1` with `meta_conv` to handle
    additional channel.
    """

    def __init__(self, sep_cfg=list(), **kwargs) -> None:
        super().__init__(**kwargs)
        self.sep_cfg = sep_cfg
        if 'backbone' in self.sep_cfg:
            layers = [self.layer1, self.layer2, self.layer3, self.layer4]
            for layers_id in range(self.frozen_stages, self.num_stages):
                for block_id, block in enumerate(layers[layers_id].children()):
                    conv_id = 0
                    for layer in block.children():
                        if layer.__class__.__name__ == 'Conv2d':
                            conv_id += 1
                            self._modules['layer{}'.format(layers_id + 1)]._modules['{}'.format(block_id)]._modules['conv{}'.format(conv_id)] = \
                                CombLohaModule('layer{}.{}.conv{}'.format(layers_id + 1, block_id, conv_id), layer, lora_dim=64)
                        elif layer.__class__.__name__ == 'Sequential':
                            self._modules['layer{}'.format(layers_id + 1)]._modules['{}'.format(block_id)]._modules['downsample']._modules['0'] = \
                                CombLohaModule('layer{}.{}.conv{}'.format(layers_id + 1, block_id, conv_id), layer[0], lora_dim=64)
                            conv_id += 1
    
    def reset_input(self, x):
        if 'backbone' in self.sep_cfg:
            x = x.repeat(2, 1, 1, 1)
        return x
    
    def forward(self, x: Tensor, use_meta_conv: bool = False) -> Tuple[Tensor]:
        """Forward function.

        When input with shape (N, 3, H, W) from images, the network will use
        `conv1` as regular ResNet. When input with shape (N, 4, H, W) from
        (image + mask) the network will replace `conv1` with `meta_conv` to
        handle additional channel.

        Args:
            x (Tensor): Tensor with shape (N, 3, H, W) from images
                or (N, 4, H, W) from (images + masks).
            use_meta_conv (bool): If set True, forward input tensor with
                `meta_conv` which require tensor with shape (N, 4, H, W).
                Otherwise, forward input tensor with `conv1` which require
                tensor with shape (N, 3, H, W). Default: False.

        Returns:
            tuple[Tensor]: Tuple of features, each item with
                shape (N, C, H, W).
        """
        x = self.reset_input(x)
        if use_meta_conv:
            x = self.meta_conv(x)
        else:
            x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
