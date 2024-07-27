# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import Upsample, resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead

import torch
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
import torch.nn.functional as F
from ..builder import build_loss
from ..losses import accuracy

@HEADS.register_module()
class FPN_FSDHead(BaseDecodeHead):
    """Panoptic Feature Pyramid Networks.

    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, feature_strides, bg_idx=[0], **kwargs):
        super(FPN_FSDHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.fam_module = FAM(in_channels=128, out_channels=128)
        self.scale_heads = nn.ModuleList()
        self.bg_idx = bg_idx
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
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
                if feature_strides[i] != feature_strides[0]:
                    scale_head.append(
                        Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=self.align_corners))
            self.scale_heads.append(nn.Sequential(*scale_head))

    def forward(self, inputs):

        x = self._transform_inputs(inputs)

        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.feature_strides)):
            # non inplace
            output = output + resize(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        # from matplotlib import pyplot as plt
        # import torch
        # b, c, h, w = output.shape
        # output = output.permute(0, 2, 3, 1).contiguous().view(b, -1, c)

        # for i in range(output.shape[1]//50):
        #     plt.imsave(f'vis_img20240304/{i+3}_bg.png',-torch.matmul(output[:,i*50],output.permute(0,2,1)).contiguous().view(b,-1, h, w)[0][0].cpu().numpy(),cmap='jet')
        #     plt.imsave(f'vis_img20240304/{i+3}_fg.png',torch.matmul(output[:,i*50],output.permute(0,2,1)).contiguous().view(b,-1, h, w)[0][0].cpu().numpy(),cmap='jet')
        fg_output, output = self.fam_module(output)

        output = self.cls_seg(output)
        return fg_output, output


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
        fg_output, seg_logits = self.forward(inputs)
        # print(torch.unique(gt_semantic_seg))
        
        losses = self.losses(fg_output, seg_logits, gt_semantic_seg)

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
            Ten: Output segmentation map.
        """
        # return self.forward(inputs)[0], self.forward(inputs)[1]
        result = self.forward(inputs)
        return result[1]

    def losses(self, fg_output, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        fg_output = resize(
            input=fg_output,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)
        bg_idx_tensor = torch.tensor(self.bg_idx).to(seg_label.device)   # 如果bg_idx不是张量，将其转换为张量

        mask = torch.isin(seg_label, bg_idx_tensor) # 创建一个布尔张量，其中labels中的元素在bg_idx中的位置为True

        seg_label = torch.where(mask, torch.zeros_like(seg_label), seg_label)
        fg_mask = torch.ones_like(seg_label)
        fg_mask[mask] = 0
        fg_label = torch.where(seg_label == 0, torch.zeros_like(seg_label), torch.ones_like(seg_label))
        # import pdb; pdb.set_trace()
        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode

        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
        # import pdb; pdb.set_trace()
        
        loss['fg_loss']= F.binary_cross_entropy(fg_output.squeeze(1), fg_label.float())
        loss['acc_seg'] = accuracy(
            seg_logit, seg_label, ignore_index=self.ignore_index)
        loss['acc_seg_fg'] = accuracy(
            fg_output, fg_label, ignore_index=self.ignore_index)
        
        return loss


class FAM(nn.Module):
    def __init__(self, in_channels=128, out_channels=128):
        super(FAM, self).__init__()
        self.in_channels = in_channels
        self.key_channels =out_channels

        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),nn.BatchNorm2d(out_channels),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False)
        )
        
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(), nn.BatchNorm2d(out_channels),
            nn.Conv2d(in_channels=self.key_channels, out_channels=1,
                kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.conv = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1, padding=0), nn.ReLU(),
                                            nn.BatchNorm2d(out_channels))
        
    def forward(self, fpn_feature):

        ## course output:B*K*H/8*W/8 , c4: B*C*H/8*W/8 , N=H/8*W/8
        output_feature = self.f_pixel(fpn_feature)
        fg_output = self.f_object(output_feature)
        
        fg_output = torch.sigmoid(fg_output)

        fam_feature = torch.multiply(fpn_feature, fg_output)
        fam_feature = self.conv(fam_feature) + fpn_feature
        return fg_output, fam_feature
