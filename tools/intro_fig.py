import cv2
import os
import numpy as np

# tfa_base_ans_dir = '/home/jianghx/data/DIOR/Output/base_pretraining/split2/JPEGImages'
# tfa_fewshot_ans_dir = '/home/jianghx/data/DIOR/Output/novel_baseline/split2/JPEGImages'
# osr_fewshot_ans_dir = '/home/jianghx/data/DIOR/Output/novel_BOSS/split2/JPEGImages'
# gt_dir = '/home/jianghx/data/DIOR/Output/gt'
# out_ans_dir = '/home/jianghx/data/DIOR/Output/check_ans/split2/JPEGImages'

tfa_base_ans_dir = '/home/jianghx/data/nwpu/Output/base_pretraining/split1/JPEGImages'
tfa_fewshot_ans_dir = '/home/jianghx/data/nwpu/Output/novel_baseline/split1/JPEGImages'
osr_fewshot_ans_dir = '/home/jianghx/data/nwpu/Output/novel_BOSS/split1/JPEGImages'
gt_dir = '/home/jianghx/data/nwpu/Output/gt'
out_ans_dir = '/home/jianghx/data/nwpu/Output/check_ans/split1/JPEGImages'

osr_fewshot_img_list = os.listdir(osr_fewshot_ans_dir)
tfa_fewshot_img_list = os.listdir(tfa_fewshot_ans_dir)
tfa_base_img_list = os.listdir(tfa_base_ans_dir)
gt_img_list = os.listdir(gt_dir)
os.makedirs(out_ans_dir, exist_ok=True)

for img in osr_fewshot_img_list:
    if (img in tfa_fewshot_img_list) * (img in tfa_base_img_list) * (img in  gt_img_list):
        osr_fewshot_img = cv2.imread(os.path.join(osr_fewshot_ans_dir, img))
        tfa_fewshot_img = cv2.imread(os.path.join(tfa_fewshot_ans_dir, img))
        tfa_base_img = cv2.imread(os.path.join(tfa_base_ans_dir, img))
        gt_img = cv2.imread(os.path.join(gt_dir, img))
        cv2.imwrite(os.path.join(out_ans_dir, img), np.hstack([gt_img, tfa_base_img, tfa_fewshot_img, osr_fewshot_img]))