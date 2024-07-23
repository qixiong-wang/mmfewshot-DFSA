import numpy as np
import cv2
from mmcv.ops import batched_nms
import torch
import os.path as osp
import xml.etree.ElementTree as ET
import os
classes = ('airport', 'basketballcourt', 'bridge', 'chimney', 
                       'dam', 'Expressway-Service-area', 'Expressway-toll-station', 
                       'golffield', 'groundtrackfield', 'harbor','overpass', 
                       'ship', 'stadium', 'storagetank', 'vehicle', 
                       'airplane','baseballfield', 'tenniscourt', 'trainstation', 'windmill')
def _get_xml_ann_info(img_id: str):
        """Get annotation from XML file by img_id.

        Args:
            dataset_year (str): Year of voc dataset. Options are
                'VOC2007', 'VOC2012'
            img_id (str): Id of image.
            classes (list[str] | None): Specific classes to load form
                xml file. If set to None, it will use classes of whole
                dataset. Default: None.

        Returns:
            dict: Annotation info of specified id with specified class.
        """
        classes = ('airport', 'basketballcourt', 'bridge', 'chimney', 
                       'dam', 'Expressway-Service-area', 'Expressway-toll-station', 
                       'golffield', 'groundtrackfield', 'harbor','overpass', 
                       'ship', 'stadium', 'storagetank', 'vehicle', 
                       'airplane','baseballfield', 'tenniscourt', 'trainstation', 'windmill')
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []

        cat2label = {cat: i for i, cat in enumerate(classes)}
        xml_path = osp.join('../data/DIOR', 'Annotations', 'Horizontal Bounding Boxes',
                            f'{img_id}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in classes:
                continue
            label = cat2label[name]
            difficult = obj.find('difficult')
            difficult = 0 if difficult is None else int(difficult.text)
            bnd_box = obj.find('bndbox')

            # It should be noted that in the original mmdet implementation,
            # the four coordinates are reduced by 1 when the annotation
            # is parsed. Here we following detectron2, only xmin and ymin
            # will be reduced by 1 during training. The groundtruth used for
            # evaluation or testing keep consistent with original xml
            # annotation file and the xmin and ymin of prediction results
            # will add 1 for inverse of data loading logic.
            bbox = [
                int(float(bnd_box.find('xmin').text)),
                int(float(bnd_box.find('ymin').text)),
                int(float(bnd_box.find('xmax').text)),
                int(float(bnd_box.find('ymax').text))
            ]
            ignore = False
            if difficult or ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                
                # if bbox[-1] - 
                labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2)
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2)
            labels_ignore = np.array(labels_ignore)
        ann_info = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann_info
    
def IoU(box1, box2):
    """
    :param box1: list in format [xmin1, ymin1, xmax1, ymax1]
    :param box2:  list in format [xmin2, ymin2, xamx2, ymax2]
    :return:    returns IoU ratio (intersection over union) of two boxes
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    x_overlap = max(0, min(xmax1, xmax2) - max(xmin1, xmin2))
    y_overlap = max(0, min(ymax1, ymax2) - max(ymin1, ymin2))
    intersection = x_overlap * y_overlap
    union = (xmax1 - xmin1) * (ymax1 - ymin1) + (xmax2 - xmin2) * (ymax2 - ymin2) - intersection
    return float(intersection) / union


colors = [(58, 155, 58), (139, 58, 58), (211, 139, 58), 
          (58, 211, 211), (58, 58, 139), (148, 0, 211), 
          (211, 0, 148), (211, 148, 0), 
          (0, 148, 211), (0, 58, 148), 
          (0, 211, 211), (139, 139, 139)]
for png_name in os.listdir('../data/DIOR/JPEGImages-test'):
    # if png_name != '00008.jpg':
    #     continue
    png = cv2.imread(os.path.join('../data/DIOR/JPEGImages-test', png_name))
    label = _get_xml_ann_info(png_name.split('.')[0])
    i = 0
    for gt_bbox, label in zip(label['bboxes'], label['labels']):
        
        gt_bbox = gt_bbox.astype('int')
        
        label_gt = classes[label]
        labelSize = cv2.getTextSize(label_gt + '0', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(png, (int(gt_bbox[0]), int(gt_bbox[1])), (int(gt_bbox[2]), int(gt_bbox[3])), 
                      color=tuple([i - 20 for i in colors[9]]), thickness=2)
        cv2.putText(png, label_gt,
                    (gt_bbox[0], gt_bbox[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    thickness=1
                    )
        
    cv2.imwrite('/home/jianghx/data/DIOR/Output/gt/{}'.format(png_name), png)