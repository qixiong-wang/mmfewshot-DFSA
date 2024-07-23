import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

grad_path = 'work_dirs/tfa_bk/tfa_r101_fpn_dior-split1_10shot-fine-tuning-bb/grad/grad.txt'
grad = np.loadtxt(grad_path)
grad = grad[(grad[:, 1] == 3) + (grad[:, 1] == 0) + (grad[:, 1] == 1)]
grad[:, 0] = np.cos(grad[:, 0] / 180 * np.pi)
grad = grad[np.abs(grad[:, 0]) >= 0.01]
names = {
    0: 'fc_cls\nweight', 
    1: 'box_fc0\nweight', 
    # 2: 'box_fc0\nbias',
    3: 'box_fc1\nweight', 
    # 4: 'box_fc1\nbias'
    }
def numbers_layers(num):
    return names[num]

vec_angles = grad[grad[:, 0] != 90, 0].tolist()
Layers = grad[grad[:, 0] != 90, 1].tolist()


Layers = list(map(lambda num: names[num], Layers))

ax = plt.figure(figsize=(8, 6))
#设置风格
sns.set(style="whitegrid")
# 构建数据
data = pd.DataFrame({
    "Gradient Direction Angle": vec_angles,
    "Network Layers": Layers
})
"""
案例5
绘制水平方向的分类散点图

可以对案例1和案例5 进行比较
"""
# sns.histplot(data=data, x="Gradient Direction Angle", hue="Network Layers", kde=True)
# plt.show()
kde = sns.kdeplot(data=data, x="Gradient Direction Angle", hue="Network Layers", fill=True, bw_adjust=0.35, common_norm=False)
# sns.stripplot(x="Gradient Direction Angle", y="Network Layers", data=data, jitter=True,
#                 hue="Network Layers", order = ['fc_cls\nweight', 'box_fc0\nweight', 'box_fc1\nweight', 'box_fc0\nbias', 'box_fc1\nbias'], 
#                 palette=sns.color_palette("plasma_r", n_colors=5), alpha=0.5, marker='h', size=8)
plt.ylabel('Count')
plt.title('')
plt.savefig('grad.pdf', dpi=400, bbox_inches='tight') 
plt.cla()