# Copyright (c) OpenMMLab. All rights reserved.

import mmcv
from mmcv.utils import print_log

from ..utils import get_root_logger
from .builder import DATASETS
from .custom import CustomDataset
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


@DATASETS.register_module()
class FewShotSSeg_iSAIDDataset(CustomDataset):
    """ iSAID: A Large-scale Dataset for Instance Segmentation in Aerial Images
    In segmentation map annotation for iSAID dataset, which is included
    in 16 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '_manual1.png'.
    """


    # PALETTE = [[0, 0, 0], [0, 0, 63], [0, 63, 63], [0, 63, 0], [0, 63, 127],
    #            [0, 63, 191], [0, 63, 255], [0, 127, 63], [0, 127, 127],
    #            [0, 0, 127], [0, 0, 191], [0, 0, 255], [0, 191, 127],
    #            [0, 127, 191], [0, 127, 255], [0, 100, 155]]

    PALETTE =[
    [0, 0, 0],        
    [34, 177, 76],  
    [160, 160, 255],  
    [255, 160, 160], 
    [160, 255, 160],   
    [0, 63, 191],  
    [160, 50, 50], 
    [255, 160, 255],   
    [0, 127, 127],  
    [255, 255, 50],
    [0, 0, 191],
    [185, 255, 185],
    [0, 191, 127], 
    [0, 127, 191], 
    [0, 127, 255],  
    [0, 100, 155]
    ]
    def __init__(self,
                num_novel_shots=None,
                num_base_shots=None,
                img_suffix='.png',
                seg_map_suffix='.png',
                classes = None,
                ann_shot_filter=None,
                **kwargs):
        
        # assert self.file_client.exists(self.img_dir)
        self.SPLIT = dict(
        ALL_CLASSES_SPLIT1=('background', 'ship', 'store_tank', 'baseball_diamond',
               'tennis_court', 'basketball_court', 'Ground_Track_Field',
               'Bridge', 'Large_Vehicle', 'Small_Vehicle', 'Helicopter',
               'Swimming_pool', 'Roundabout', 'Soccer_ball_field', 'plane',
               'Harbor'),
        
        ALL_CLASSES_SPLIT2=('background', 'ship', 'store_tank', 'baseball_diamond',
               'tennis_court', 'basketball_court', 'Ground_Track_Field',
               'Bridge', 'Large_Vehicle', 'Small_Vehicle', 'Helicopter',
               'Swimming_pool', 'Roundabout', 'Soccer_ball_field', 'plane',
               'Harbor'),
        
        ALL_CLASSES_SPLIT3=('background', 'ship', 'store_tank', 'baseball_diamond',
               'tennis_court', 'basketball_court', 'Ground_Track_Field',
               'Bridge', 'Large_Vehicle', 'Small_Vehicle', 'Helicopter',
               'Swimming_pool', 'Roundabout', 'Soccer_ball_field', 'plane',
               'Harbor'),
        
        NOVEL_CLASSES_SPLIT1=('ship', 'store_tank', 'baseball_diamond','tennis_court', 'basketball_court', ),
        
        NOVEL_CLASSES_SPLIT2=('Ground_Track_Field', 'Bridge', 'Large_Vehicle', 'Small_Vehicle', 'Helicopter'),
        
        NOVEL_CLASSES_SPLIT3=('Swimming_pool', 'Roundabout', 'Soccer_ball_field', 'plane', 'Harbor'),
        
        BASE_CLASSES_SPLIT1=('background', 'Ground_Track_Field', 'Bridge', 'Large_Vehicle', 'Small_Vehicle', 'Helicopter',
               'Swimming_pool', 'Roundabout', 'Soccer_ball_field', 'plane', 'Harbor'),
        
        BASE_CLASSES_SPLIT2=('background', 'ship', 'store_tank', 'baseball_diamond', 'tennis_court', 'basketball_court',
                             'Swimming_pool', 'Roundabout', 'Soccer_ball_field', 'plane', 'Harbor'),
        
        BASE_CLASSES_SPLIT3=('background', 'Ground_Track_Field', 'Bridge', 'Large_Vehicle', 'Small_Vehicle', 'Helicopter',
               'Swimming_pool', 'Roundabout', 'Soccer_ball_field', 'plane', 'Harbor'),
        )
        
        
        self.num_novel_shots = num_novel_shots
        self.num_base_shots = num_base_shots

        self.CLASSES = self.get_classes(classes)

        super(FewShotSSeg_iSAIDDataset, self).__init__(
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
            print(len(self.img_infos))
        
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

        # if len(torch.unique(results['gt_semantic_seg'].data))==1:
        #     # print(11111111111)
        #     return None
        # print(torch.unique(results['gt_semantic_seg'].data))
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
                try:
                    class_name = self.CLASSES[label]
                    filter_instances[class_name].append(ann[:5])
                except:
                    pass
            if idx > 2000:
                break
        for class_name in ann_shot_filter.keys():
            set_instances = set(filter_instances[class_name])
            filter_instances[class_name] = list(set_instances)

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
            if img_info['ann']['seg_map'][:5] in keep_image_indices:
                new_img_infos.append(
                    dict(
                        filename=img_info['filename'],
                        ann=img_info['ann']))
        new_img_infos = new_img_infos*200

        return new_img_infos

    def load_annotations(self,
                         img_dir,
                         img_suffix,
                         ann_dir,
                         seg_map_suffix=None,
                         split=None):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    name = line.strip()
                    img_info = dict(filename=name + img_suffix)
                    if ann_dir is not None:
                        ann_name = name + '_instance_color_RGB'
                        seg_map = ann_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_img = img
                    seg_map = seg_img.replace(
                        img_suffix, '_instance_color_RGB' + seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())

        return img_infos
    
    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """
        
        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            # import time
            # time1 = time.time()
            # print('idx:', idx)
            data = self.prepare_train_img(idx)
            while data is None:
                idx = np.random.randint(0, len(self.img_infos))
                data =  self.prepare_train_img(idx)
                print('idx:', idx)
            # time2 = time.time()
            # print('data_prepare time:', time2-time1)
            return data 
            # return self.prepare_train_img(idx)
        