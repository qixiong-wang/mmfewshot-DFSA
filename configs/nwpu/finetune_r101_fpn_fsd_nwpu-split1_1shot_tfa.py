img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(896, 896), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(896, 896), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    # dict(type='PolyRandomRotate', rotate_ratio=0.5),
    dict(
        type='Normalize',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data_root = './data/NWPU_VHR_10_VOC/'
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    persistent_workers =False,
    train=dict(
        type='FewShotSSeg_NWPUDataset',
        data_root='data/NWPU_VHR_10_VOC',
        img_dir='JPEGImages',
        ann_dir='masks',
        num_novel_shots=1,
        num_base_shots=1,
        split='Main/trainval.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=(896, 896), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(896, 896), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            # dict(type='PolyRandomRotate', rotate_ratio=0.5),
            dict(
                type='Normalize',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ],
        classes='ALL_CLASSES_SPLIT1',),
    val=dict(
        type='FewShotSSeg_NWPUDataset',
        data_root='data/NWPU_VHR_10_VOC',
        img_dir='JPEGImages',
        ann_dir='masks',
        split='Main/test.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(896, 896),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[103.53, 116.28, 123.675],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes='ALL_CLASSES_SPLIT1',),
    test=dict(
        type='FewShotSSeg_NWPUDataset',
        data_root='data/NWPU_VHR_10_VOC',
        img_dir='JPEGImages',
        ann_dir='masks',
        split='Main/test.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(896, 896),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[103.53, 116.28, 123.675],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        test_mode=True,
        classes='ALL_CLASSES_SPLIT1'))
evaluation = dict(
    interval=2000,
    metric=['mIoU', 'mFscore'],
    class_splits=['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'])
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=0.001,
    step=[4000,8000])
runner = dict(type='IterBasedRunner', max_iters=10000)

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/twins/pcpvt_small_20220308-e638c41c.pth'
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 2),
        strides=(1, 2, 2, 1),
        multi_grid=(1, 2, 4),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4),
    decode_head=dict(
        type='FPN_FSDHead',
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=11,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

checkpoint_config = dict(interval=2000)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
# custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'work_dirs/nwpu/base_training/r101_fpn_fsd_nwpu-split1_base-training/iter_18000.pth'
resume_from = None
workflow = [('train', 1)]
use_infinite_sampler = False
seed = 42
work_dir = 'work_dirs/nwpu/finetune/finetune_r101_fpn_fsd_nwpu-split1_1shot_tfa-0'
gpu_ids = range(0, 2)
