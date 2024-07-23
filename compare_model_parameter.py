import torch
import torch.nn as nn
import torch.nn.functional as F

model_path1 = 'work_dirs/boss_r101_fpn_isaid-split3_1shot-fine-tuning_resize-5/iter_5000.pth'
model_path2 = 'work_dirs/tfa_r101_fpn_isaid-split3_base_training-resize-lr0.005-0/iter_80000.pth'
model1 = torch.load(model_path1)

model2 = torch.load(model_path2)
## compare the parameters of the two models
for name1, param1 in model1['state_dict'].items():
    if name1 in model2['state_dict']:
        param2 = model2['state_dict'][name1]
        if not torch.equal(param1, param2):
            print(f'Parameter {name1} is different')
        else:
            print(f'Parameter {name1} is the same')
    else:
        print(f'Parameter {name1} is not in model2')