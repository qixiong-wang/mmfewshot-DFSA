import numpy as np
from sklearn.manifold import TSNE
import torch


import pyecharts.options as opts
from pyecharts.charts import Scatter3D
# 生成一些示例数据
classes = ['airport', 'basketballcourt', 'bridge', 'chimney', 'dam', 
                        'Expressway-Service-area', 'Expressway-toll-station', 'golffield', 'groundtrackfield', 'harbor',
                        'overpass', 'ship', 'stadium', 'storagetank', 'vehicle', 
                        'airplane', 'baseballfield', 'tenniscourt', 'trainstation', 'windmill']
data = torch.load('ft_boss.pt', map_location='cuda:0').cpu()
class_id = (data[:, -1] > 14) + (data[:, -1] < 5)
# 使用 t-SNE 进行降维
tsne = TSNE(n_components=3, random_state=42)
tsne_result = tsne.fit_transform(data[:, :1024])

tsne_result = tsne_result[class_id]
data = np.concatenate([tsne_result, data[class_id, -1:].numpy()], axis=-1)

piece=[
      {'value': 0,'label': 'airport','color':'#FFFFCC'}, 
      {'value': 1, 'label': 'basketballcourt','color':'#FFFF99'},
      {'value': 2, 'label': 'bridge','color':'#FFCC99'},
      {'value': 3, 'label': 'chimney','color':'#FF9966'},
      {'value': 4, 'label': 'dam','color':'#CC9999'},
      {'value': 15, 'label': 'airplane','color':'#CCFFFF'},
      {'value': 16, 'label': 'baseballfield','color':'#99CCFF'},
      {'value': 17, 'label': 'tenniscourt','color':'#99CC99'},
      {'value': 18, 'label': 'trainstation','color':'#99CCCC'},
      {'value': 19, 'label': 'windmill','color':'#9999CC'}
    ]

(
    Scatter3D()  # bg_color="black"
    .add(
        series_name="",
        data=data.tolist(),
        xaxis3d_opts=opts.Axis3DOpts(
            name='x',
            type_="value",
            # textstyle_opts=opts.TextStyleOpts(color="#fff"),
        ),
        yaxis3d_opts=opts.Axis3DOpts(
            name='y',
            type_="value",
            # textstyle_opts=opts.TextStyleOpts(color="#fff"),
        ),
        zaxis3d_opts=opts.Axis3DOpts(
            name='z',
            type_="value",
            # textstyle_opts=opts.TextStyleOpts(color="#fff"),
        ),
        grid3d_opts=opts.Grid3DOpts(width=100, height=100, depth=100),
    )
    .set_global_opts(
        visualmap_opts=[
            opts.VisualMapOpts(
                type_="color",
                is_calculable=True,
                dimension=3,
                is_piecewise=True, 
                pieces=piece
            ),
        ]
    )
    .render("scatter3d_boss.html")
)