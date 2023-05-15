_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    type='TOOD',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='TOODHead_refine',
        num_classes=80,
        in_channels=256,
        stacked_convs=6,
        feat_channels=256,
        anchor_type='anchor_refine',
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),  # 检测[8,16,32,64,128]*8的正方形区域大小anchor
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        initial_loss_cls=dict(
            type='FocalLossWithProb',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_cls=dict(
            type='TaskAlignedFocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
    ))
# training and testing settings
train_cfg = dict(
    initial_epoch=0,
    initial_assigner=dict(type='ATSSAssigner', topk=9),
    assigner=dict(type='TaskAlignedAssigner', topk=13),
    alpha=1,
    beta=6,
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_threshold=0.6),
    max_per_img=100)
# optimizer
optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001)

# custon hooks: HeadHook is defined in mmdet/core/utils/head_hook.py
custom_hooks = [
    dict(type="HeadHook")
]

data = dict(samples_per_gpu=4,
            workers_per_gpu=2)

work_dir = './work_dirs/tood_r50_fpn_1x_coco_refine_last5epoch'
checkpoint_config = dict(interval=1, max_keep_ckpts=5, out_dir='checkpoints/tood_r50_fpn_1x_coco_refine')
# evaluation = dict(save_best='auto')
# resume_from = '/media/store1/sxw/code/tood/checkpoints/tood_r50_fpn_1x_coco_initial/tood_r50_fpn_1x_coco_initial/latest.pth'
load_from = '/root/autodl-tmp/code/checkpoints/tood_r50_fpn_1x_coco_refine/epoch_9.pth'
runner = dict(type='EpochBasedRunner', max_epochs=5)
