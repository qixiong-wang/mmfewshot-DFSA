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
from mmfewshot.mmseg.models.utils.sep_fpn_conv import SepFPNConvModule
import torch.nn.functional as F
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

    def __init__(self, feature_strides, sep_cfg, novel_idx=None, **kwargs):
        super(SepFPNHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.sep_cfg = sep_cfg
        self.novel_idx = novel_idx
        self.scale_heads = nn.ModuleList()
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            for k in range(head_length):
                if 'neck' in self.sep_cfg:
                    scale_head.append(
                        SepFPNConvModule(
                            self.in_channels[i] if k == 0 else self.channels,
                            self.channels,
                            3,
                            padding=1,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg))
                else:
                    scale_head.append(
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
            self.scale_heads.append(nn.Sequential(*scale_head))

        self.conv_hada_w_seg = nn.Conv2d(128, self.num_classes, kernel_size=1)

        # self.conv_hada_w_seg = nn.Conv2d(128*2, len(self.novel_idx)+1, kernel_size=1)
        # self.hada_w_merge = nn.Parameter(torch.zeros(2, self.num_classes))
        # self.conv_seg1 = nn.Conv2d(128, self.num_classes, kernel_size=1)
        # self.reset_layers()
        
    # def reset_layers(self):
    #     if 'head' in self.sep_cfg:
    #         import pdb
    #         pdb.set_trace()
    #         self.scale_heads = LohaModule('fpn_head.scale_heads', self.scale_heads, lora_dim=64)
            
    #         self.conv_seg = LohaModule('fpn_head.conv_seg', self.conv_seg, lora_dim=64)

    # def reset_input(self, x):
    #     if 'neck' in self.sep_cfg:
    #         src_x, lora_x = x.split(len(x) // 2)
    #     else:
    #         src_x, lora_x = x, x
    #     return src_x, lora_x
        
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
        src_output,lora_output = self.forward(inputs)
        # print(torch.unique(gt_semantic_seg))

        losses = self.losses(lora_output, gt_semantic_seg)
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
        src_output,lora_output = self.forward(inputs)
        output = src_output
        scr_prediction = torch.argmax(src_output,dim=1)
 
        bg_idx = scr_prediction==0
        output[:,:,bg_idx[0]] = lora_output[:,:,bg_idx[0]]
        return output
    
    def cls_seg(self, src_output,lora_output):
        """Classify each pixel."""
        if self.dropout is not None:
            src_output = self.dropout(src_output)
            lora_output = self.dropout(lora_output)

        src_output = self.conv_seg(src_output) ### B*16*H*W
        # scr_result = torch.argmax(F.softmax(src_output, dim=1), dim=1)
        # import cv2
        # cv2.imwrite('src_output.png', scr_result[0].cpu().numpy().astype(np.uint8)*255)
        # base_fg = (scr_result != 0).unsqueeze(1).expand_as(src_output)
        # base_bg = (scr_result == 0).unsqueeze(1).expand_as(src_output)
        # lora_output = self.conv_seg1(lora_output) ### B*16*H*W

        lora_output = self.conv_hada_w_seg(lora_output)

        # output = src_output
        # output[base_bg] = lora_output[base_bg]
        # output[base_bg] 
        # output[:,0] = lora_output[:,0]
        # output[:,self.novel_idx] = lora_output[:,1:]
        
        # lora_output = self.conv_hada_w_seg(lora_output) ### B*H*W*16
        # output = torch.cat((src_output[:,:8],lora_output[:,8:]),dim=1)
        # output = torch.cat((lora_output[:,0:1],src_output[:,1:8],lora_output[:,8:]),dim=1)

        # output = lora_output * self.hada_w_merge[0].sigmoid().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  + src_output * self.hada_w_merge[1].sigmoid().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        return src_output,lora_output


    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        # src_x, lora_x = [],[]
        # for feature in x:
        #     src_feature, lora_feature = self.reset_input(feature)
        #     src_x.append(src_feature)
        #     lora_x.append(lora_feature)
        # src_x = self._transform_inputs(src_x)
        # lora_x = self._transform_inputs(lora_x)
        # import pdb
        # pdb.set_trace()
        # src_output,lora_output = self.scale_heads[0]((src_x[0],lora_x[0]))

        output  = self.scale_heads[0](x[0])
        for i in range(1, len(self.feature_strides)):
            # non inplace
            output = output + resize(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            
            # src_output_temp, lora_output_temp = self.scale_heads[i](src_x[i], lora_x[i])
            # src_output =  src_output + resize(
            #     src_output_temp,
            #     size=src_output.shape[2:],
            #     mode='bilinear',
            #     align_corners=self.align_corners)
            # lora_output =  lora_output + resize(
            #     lora_output_temp,
            #     size=lora_output.shape[2:],
            #     mode='bilinear',
            #     align_corners=self.align_corners)
        src_output, lora_output = output.split(len(output) // 2)
        output = self.cls_seg(src_output,lora_output)

        return output

