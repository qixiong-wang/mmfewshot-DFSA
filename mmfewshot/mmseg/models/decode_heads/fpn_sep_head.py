# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import Upsample, resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
import torch
from mmfewshot.mmseg.models.utils.loha import LohaModule
from .fpn_head import FPNHead


@HEADS.register_module()
class SepFPNHead(BaseDecodeHead):
    """Panoptic Feature Pyramid Networks.

    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, feature_strides, sep_cfg, **kwargs):
        super(SepFPNHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.sep_cfg = sep_cfg
        self.scale_heads = nn.ModuleList()
        self.scale_heads_novel = nn.ModuleList()

        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            scale_head_novel = []
            for k in range(head_length):
                scale_head.append(
                    ConvModule(
                        self.in_channels[i] if k == 0 else self.channels,
                        self.channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                scale_head_novel.append(
                    ConvModule(
                        self.in_channels[i] if k == 0 else self.channels,
                        self.channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                if feature_strides[i] != feature_strides[0]:
                    scale_head.append(
                        Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=self.align_corners))
                    scale_head_novel.append(
                        Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=self.align_corners))
                    
            self.scale_heads.append(nn.Sequential(*scale_head))
            self.scale_heads_novel.append(nn.Sequential(*scale_head_novel))
            # self.scale_heads_novel = self.create_lora_layers(self.scale_heads)
            self.conv_seg_novel = nn.Conv2d(128, self.num_classes, kernel_size=1)

    # def create_lora_layers(self, model):
    #     for name, module in model.named_modules():
    #         in_features = module.in_features
    #         out_features = module.out_features
    #         self.add_module(name + "_lora_down", nn.Linear(in_features, self.rank, bias=False))
    #         self.add_module(name + "_lora_up", nn.Linear(self.rank, out_features, bias=False))

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        output_base, output_novel = self.forward(inputs)
        # print(torch.unique(gt_semantic_seg))

        losses = self.losses(output_novel, gt_semantic_seg)
        return losses
       

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        base_output,novel_output = self.forward(inputs)
        output = base_output
        scr_prediction = torch.argmax(base_output,dim=1)
 
        bg_idx = scr_prediction==0
        output[:,:,bg_idx[0]] = novel_output[:,:,bg_idx[0]]
        return output
    

    def forward(self, inputs):
        
        x = self._transform_inputs(inputs[0])
        x_novel = self._transform_inputs(inputs[1])

        output  = self.scale_heads[0](x[0])
        output_novel = self.scale_heads_novel[0](x_novel[0]) 
        # +  self.scale_heads_novel_lora_up[0](self.scale_heads_novel_lora_down[0](x_novel[0]))
        for i in range(1, len(self.feature_strides)):
            output = output + resize(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            
            output_novel =  output_novel + resize(
                self.scale_heads_novel[i](x_novel[i]),
                # self.scale_heads_novel[i](x[i]) + self.scale_heads_novel_lora_up[i](self.scale_heads_novel_lora_down[i](x[i])),
                size=output_novel.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            
        output = self.cls_seg(output)
        output_novel = self.conv_seg_novel(output_novel)

        return output, output_novel

