# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import xml.etree.ElementTree as ET
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union
import os
import mmcv
import numpy as np
from mmcv.utils import print_log
import cv2
from .builder import DATASETS
# from .base import FewShotSeg_BaseUDataset

import torch

from .custom import CustomDataset
# pre-defined classes split for few shot setting



@DATASETS.register_module()
class FewShotSSeg_NWPUDataset(CustomDataset):
    """VOC dataset for few shot detection.

    Args:
        classes (str | Sequence[str]): Classes for model training and
            provide fixed label for each class. When classes is string,
            it will load pre-defined classes in `FewShotVOCDataset`.
            For example: 'NOVEL_CLASSES_SPLIT1'.
        num_novel_shots (int | None): Max number of instances used for each
            novel class. If is None, all annotation will be used.
            Default: None.
        num_base_shots (int | None): Max number of instances used
            for each base class. When it is None, all annotations
            will be used. Default: None.
        ann_shot_filter (dict | None): Used to specify the class and the
            corresponding maximum number of instances when loading
            the annotation file. For example: {'dog': 10, 'person': 5}.
            If set it as None, `ann_shot_filter` will be
            created according to `num_novel_shots` and `num_base_shots`.
            Default: None.
        use_difficult (bool): Whether use the difficult annotation or not.
            Default: False.
        min_bbox_area (int | float | None):  Filter images with bbox whose
            area smaller `min_bbox_area`. If set to None, skip
            this filter. Default: None.
        dataset_name (str | None): Name of dataset to display. For example:
            'train dataset' or 'query dataset'. Default: None.
        test_mode (bool): If set True, annotation will not be loaded.
            Default: False.
        coordinate_offset (list[int]): The bbox annotation will add the
            coordinate offsets which corresponds to [x_min, y_min, x_max,
            y_max] during training. For testing, the gt annotation will
            not be changed while the predict results will minus the
            coordinate offsets to inverse data loading logic in training.
            Default: [-1, -1, 0, 0].
    """
    # PALETTE=[[0, 0, 0], [102, 179, 92], [14, 106, 71], [188, 20, 102], [121, 210, 214], [74, 202, 87], [116, 99, 103], [151, 130, 149], [52, 1, 87], [235, 157, 37], [129, 191, 187]]
    # PALETTE = [
    # [0, 0, 0],        # Black
    # [128, 128, 255],  # Light Blue
    # [255, 128, 128],  # Light Red
    # [128, 255, 128],  # Light Green
    # [255, 255, 128],  # Light Yellow
    # [128, 255, 255],  # Light Cyan
    # [255, 128, 255],  # Light Magenta
    # [192, 192, 192],  # Silver
    # [255, 165, 0],    # Orange
    # [173, 216, 230],  # Light Blue (Baby Blue)
    # [144, 238, 144]]  # Light Green (Light Green)]       # Deep Green
    PALETTE = [
    [0, 0, 0],        # Black
    [128, 128, 255],  # Light Blue
    [255, 128, 128],  # Light Red
    [191, 63, 0],  
    [255, 255, 128],  # Light Yellow
    [128, 255, 255],  # Light Cyan
    [255, 128, 255],  # Light Magenta
    [192, 192, 192],  # Silver
    [255, 165, 0],    # Orange
    [173, 216, 230],  # Light Blue (Baby Blue)
    [144, 238, 144]]  # Light Green (Light Green)]       # Deep Green
    for i in range(len(PALETTE)):
        PALETTE[i] =  PALETTE[i][::-1]
    def __init__(self,
                num_novel_shots=None,
                num_base_shots=None,
                img_suffix='.jpg',
                seg_map_suffix='.png',
                classes = None,
                ann_shot_filter=None,
                **kwargs):
        
        self.SPLIT = dict(
            ALL_CLASSES_SPLIT1=('background', 'bridge', 'harbor', 'ground track field', 'ship', 'storage tank', 
                        'basketball court', 'vehicle', 'airplane', 'baseball diamond', 'tennis court'),
            NOVEL_CLASSES_SPLIT1=('airplane', 'baseball diamond', 'tennis court'),
            # BASE_CLASSES_SPLIT1=('bridge', 'harbor', 'ground track field', 'ship', 'storage tank',
            #                     'basketball court', 'vehicle')
            BASE_CLASSES_SPLIT1=('background','bridge', 'harbor', 'ground track field', 'ship', 'storage tank', 
                        'basketball court', 'vehicle'),
        )
        
        self.num_novel_shots = num_novel_shots
        self.num_base_shots = num_base_shots

        self.CLASSES = self.get_classes(classes)
        super(FewShotSSeg_NWPUDataset, self).__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs)
        

        if ann_shot_filter is None:
            # configure ann_shot_filter by num_novel_shots and num_base_shots
            if num_novel_shots is not None or num_base_shots is not None:
                ann_shot_filter = self._create_ann_shot_filter()
        else:
            assert num_novel_shots is None and num_base_shots is None, \
                f'{self.dataset_name}: can not config ann_shot_filter and ' \
                f'num_novel_shots/num_base_shots at the same time.'
            
        if ann_shot_filter is not None:
            if isinstance(ann_shot_filter, dict):
                for class_name in list(ann_shot_filter.keys()):
                    assert class_name in self.CLASSES, \
                        f'{self.dataset_name} : class ' \
                        f'{class_name} in ann_shot_filter not in CLASSES.'
            else:
                raise TypeError('ann_shot_filter only support dict')
            self.ann_shot_filter = ann_shot_filter
            self.img_infos = self._filter_annotations(
                self.img_infos, self.ann_shot_filter)
            
        
    def _create_ann_shot_filter(self) -> Dict[str, int]:
        """Generate `ann_shot_filter` for novel and base classes.

        Returns:
            dict[str, int]: The number of shots to keep for each class.
        """
        ann_shot_filter = {}
        if self.num_novel_shots is not None:
            for class_name in self.SPLIT[
                    f'NOVEL_CLASSES_SPLIT{self.split_id}']:
                ann_shot_filter[class_name] = self.num_novel_shots
        if self.num_base_shots is not None:
            for class_name in self.SPLIT[f'BASE_CLASSES_SPLIT{self.split_id}']:
                ann_shot_filter[class_name] = self.num_base_shots
        return ann_shot_filter
    

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        # print(img_info, ann_info)
        results = dict(img_info=img_info, ann_info=ann_info)

        self.pre_pipeline(results)
        results = self.pipeline(results)

        # results['gt_semantic_seg'].data = torch.where(results['gt_semantic_seg'].data>7,255, results['gt_semantic_seg'].data)

        return results
    
    

    def get_classes(self, classes: Union[str, Sequence[str]]) -> List[str]:
        """Get class names.

        It supports to load pre-defined classes splits.
        The pre-defined classes splits are:
        ['ALL_CLASSES_SPLIT1', 'ALL_CLASSES_SPLIT2', 'ALL_CLASSES_SPLIT3',
         'BASE_CLASSES_SPLIT1', 'BASE_CLASSES_SPLIT2', 'BASE_CLASSES_SPLIT3',
         'NOVEL_CLASSES_SPLIT1','NOVEL_CLASSES_SPLIT2','NOVEL_CLASSES_SPLIT3']

        Args:
            classes (str | Sequence[str]): Classes for model training and
                provide fixed label for each class. When classes is string,
                it will load pre-defined classes in `FewShotVOCDataset`.
                For example: 'NOVEL_CLASSES_SPLIT1'.

        Returns:
            list[str]: List of class names.
        """
        # configure few shot classes setting
        if isinstance(classes, str):
            assert classes in self.SPLIT.keys(
            ), f'{self.dataset_name}: not a pre-defined classes or ' \
               f'split in VOC_SPLIT'
            class_names = self.SPLIT[classes]
            if 'BASE_CLASSES' in classes:

                assert self.num_novel_shots is None, \
                    f'{self.dataset_name}: BASE_CLASSES do not have ' \
                    f'novel instances.'
            elif 'NOVEL_CLASSES' in classes:
                assert self.num_base_shots is None, \
                    f'{self.dataset_name}: NOVEL_CLASSES do not have ' \
                    f'base instances.'
            self.split_id = int(classes[-1])
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')
        return class_names


    def _filter_annotations(self, img_infos: List[Dict],
                            ann_shot_filter: Dict) -> List[Dict]:
        """Filter out extra annotations of specific class, while annotations of
        classes not in filter remain unchanged and the ignored annotations will
        be removed.

        Args:
            img_infos (list[dict]): Annotation infos.
            ann_shot_filter (dict): Specific which class and how many
                instances of each class to load from annotation file.
                For example: {'dog': 10, 'cat': 10, 'person': 5}

        Returns:
            list[dict]: Annotation infos where number of specified class
                shots less than or equal to predefined number.
        """

        if ann_shot_filter is None:
            return img_infos
        # build instance indices of (img_id, gt_idx)
        filter_instances = {key: [] for key in ann_shot_filter.keys()}
        keep_image_indices = []
        for idx, img_info in enumerate(img_infos):
            ann = img_info['ann']['seg_map']
            ann_label = np.unique(cv2.imread(os.path.join(self.ann_dir,ann)))
            for label in ann_label:
                # if label==0:
                    # continue
                class_name = self.CLASSES[label]
                filter_instances[class_name].append(ann)

        # filter extra shots
        for class_name in ann_shot_filter.keys():
            num_shots = ann_shot_filter[class_name]
            image_indices = filter_instances[class_name]
            if num_shots == 0:
                continue
            # random sample from all instances
            if class_name=='background':
                continue
            random_select = np.random.choice(len(image_indices), num_shots, replace=False)
            keep_image_indices += [image_indices[i] for i in random_select]
            # number of available shots less than the predefined number,
            # which may cause the performance degradation

        # keep the selected annotations and remove the undesired annotations
        new_img_infos = []

        for idx, img_info in enumerate(img_infos):
            if img_info['ann']['seg_map'] in keep_image_indices:
                new_img_infos.append(
                    dict(
                        filename=img_info['filename'],
                        ann=img_info['ann']))
        new_img_infos = new_img_infos*200
        return new_img_infos