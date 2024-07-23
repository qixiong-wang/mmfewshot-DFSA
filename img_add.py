import numpy as np
import os
import cv2

data_dir = 'crop_img_output'
image_name = '8_img.png'
# data_dir = 'refer/data/images/mscoco/images/train2014/'
# image_name = 'COCO_train2014_000000022166.jpg'
img = cv2.imread(os.path.join(data_dir,image_name))
gt = np.zeros_like(img)
gt = cv2.imread('crop_img_output/8_mask1.png')

# gt[:,1000:1400] = cv2.imread('vis_output_cityscape/frankfurt_000001_013016_leftImg8bit.pngpersons in front.png')[:,700:1100]
# gt[:,300:1900] = cv2.imread('vis_output_cityscape/frankfurt_000001_013016_leftImg8bit.pngpersons in front.png')
# gt[:,1000:1400] = cv2.imread('vis_output_cityscape_refcoco/frankfurt_000001_013016_leftImg8bit.pngPersons in front.png')[:,1000:1400] 

pixels = gt == 255
target_color = np.zeros_like(gt)
target_color[pixels[:,:,0]] = np.array([5,100,248])
# target_color[pixels[:,:,0]] = np.array([248,5,5])


img[np.where(gt==255)] = img[np.where(gt==255)]*0.7+target_color[np.where(gt==255)]*0.3
img[np.where(gt!=255)]= img[np.where(gt!=255)]*0.9+target_color[np.where(gt!=255)]*0.1
# img= img*0.7+target_color*0.3
cv2.imwrite('crop_img_output/8_mask2.png',img)
# cv2.imwrite('vis_output_cityscape_zeroshot_lavt/frankfurt_000000_003357_leftImg8bit_pred_add_Cars in picturelavt.png',img)

# cv2.imwrite('vis_output_cityscape/hamburg_000000_060215_leftImg8bit.pngred car and yellow truck_0.8.png',img)