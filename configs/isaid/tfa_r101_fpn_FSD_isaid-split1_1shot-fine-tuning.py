img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    persistent_workers =False,
    train=dict(
        type='FewShotSSeg_iSAIDDataset',
        data_root='data/iSAID/converted',
        img_dir='img_dir/train',
        ann_dir='ann_dir/train',
        num_novel_shots=1,
        num_base_shots=1,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=(896, 896), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(896, 896), cat_max_ratio=0.75)
            ,
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            # dict(type='PolyRandomRotate', rotate_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ],
        classes='ALL_CLASSES_SPLIT1',),
    val=dict(
        type='FewShotSSeg_iSAIDDataset',
        data_root='data/iSAID/converted',
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=None,
                img_ratios=[1.0],
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes='ALL_CLASSES_SPLIT1'),
    test=dict(
        type='FewShotSSeg_iSAIDDataset',
        data_root='data/iSAID/converted',
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=None,
                img_ratios=[1.0],
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        test_mode=True,
        classes='ALL_CLASSES_SPLIT1'))

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/twins/pcpvt_small_20220308-e638c41c.pth'
norm_cfg = dict(type='SyncBN', requires_grad=True)
sep_cfg = ['neck', 'head']
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet101_v1c',
    # frozen_parameters=[
    # 'backbone', 
    # 'neck', 
    # 'decode_head',
    #     # 'roi_head.bbox_head.shared_fcs'
    # ],
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
        # sep_cfg = sep_cfg,
        # novel_idx = [11,12,13,14,15],
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=16,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(896, 896), stride=(400, 400)))

checkpoint_config = dict(interval=2000)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = None
workflow = [('train', 1)]
use_infinite_sampler = False
seed = 42
# load_from = 'work_dirs/isaid/base_training/tfa_r101_fpn_isaid-split1_base_training-resize-1gpu-16w-bs8-lr0.005-0/base_model_random_init_nwpu_split1_decode_head.pth'
load_from = 'work_dirs/isaid/base_training/split1/r101_fpn_fsd_isaid-split1_base-training-0/iter_80000.pth'
evaluation = dict(interval=2000, metric=['mIoU', 'mFscore'])
optimizer = dict(
    type='AdamW',
    lr=1e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
optimizer_config = dict(grad_clip=None)
# optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=10000)

work_dir = 'work_dirs/isaid/finetune/split1/configs/isaid/DFSA_r101_fpn_isaid-split1_1shot-fine-tuning'
gpu_ids = range(0, 2)
