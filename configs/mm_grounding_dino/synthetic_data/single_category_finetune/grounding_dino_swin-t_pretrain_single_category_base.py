_base_ = '../../grounding_dino_swin-t_pretrain_base.py'

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0004,
                   weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0.1),
        }
    ),
)

model = dict(
    test_cfg=dict(max_per_img=30)
)

data_root = 'data/coco/'

# pipeline for testing
test_pipeline = [
    dict(
        type='LoadImageFromFile', backend_args=None,
        imdecode_backend='pillow'),
    dict(
        type='FixScaleResize',
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id', 'img_path', 'ori_shape', 'img_shape',
            'scale_factor', 'text', 'custom_entities',
            'tokens_positive'
        )
    )
]

# single-category dataset & evaluator
val_dataset_single_cat = dict(
    type='SingleCategoryRefDataset',
    data_root=data_root,
    ann_file='mdetr_annotations/single_category_ref_4_19.json',
    data_prefix=dict(img='single_category_ref/'),
    test_mode=True,
    return_classes=True,
    filter_cfg=dict(filter_empty_gt=False, cat_ids=[1]),
    pipeline=test_pipeline,
)

val_evaluator_single_cat = dict(
    type='SingleCategoryRefMetric',
    ann_file=data_root + 'mdetr_annotations/single_category_ref_4_19.json',
    iou_thrs=0.5
)

# override default dataloader & evaluator
val_dataloader = dict(
    dataset=dict(_delete_=True, **val_dataset_single_cat)
)
test_dataloader = val_dataloader

val_evaluator = val_evaluator_single_cat
test_evaluator = val_evaluator
