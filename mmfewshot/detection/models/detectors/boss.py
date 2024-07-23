# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.two_stage import TwoStageDetector
import numpy as np

@DETECTORS.register_module()
class BOSS(TwoStageDetector):
    """Implementation of `BOSS`_"""
    def test_params(self, img, proposals=None, rescale=False):
        """Test without augmentation."""
        input = img[0].cuda()
        tmp_img_metas = [{
                'filename': '../data/DIOR/JPEGIma.../11726.jpg', 
                'ori_filename': 'JPEGImages/11726.jpg', 
                'ori_shape': (800, 800, 3), 
                'img_shape': (800, 800, 3), 
                'pad_shape': (800, 800, 3), 
                'scale_factor': np.array([1.0, 1.0, 1.0, 1.0]), 
                'flip': False, 
                'flip_direction': None, 
                'img_norm_cfg': {
                    'mean': np.array([103.53, 116.28, 123.675]), 
                    'std': np.array([1.0, 1.0, 1.0]), 
                    'to_rgb': False}
                }]
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(input)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas=tmp_img_metas)
        else:
            proposal_list = proposals
        num_rois_per_img, num_bbox_per_img, num_score_per_img, rois, proposals = self.roi_head.reset_test_input(proposal_list, tmp_img_metas)
        outs = self.roi_head._bbox_forward(x, rois)
        return outs