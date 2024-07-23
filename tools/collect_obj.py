import numpy as np
objs = np.loadtxt('obj_record.txt')
gt_nums = objs[:, :5]
pre_nums = objs[:, 5:]
recall_pro = pre_nums.sum(0) / gt_nums.sum(0)
print(objs)