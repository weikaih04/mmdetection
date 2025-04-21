_base_ = '../grounding_dino_swin-t_pretrain_obj365.py'

model = dict(
    test_cfg=dict(max_per_img=30)
)

data_root = 'data/coco/'

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

# ---- single‐category dataset & evaluator ----
# here we take the RefCOCO_TestA split but only keep category_id=1
val_dataset_single_cat = dict(
    type='MDETRStyleRefCocoDataset',
    data_root=data_root,
    ann_file='mdetr_annotations/single_category_ref_4_19.json',
    data_prefix=dict(img='single_category_ref/'),
    test_mode=True,
    return_classes=True,
    # filter_cfg will drop all samples whose GT category_id is not in [1]
    filter_cfg=dict(filter_empty_gt=False, cat_ids=[1]),
    pipeline=test_pipeline,
    backend_args=None
)

val_dataset_single_cat = dict(
    type='SingleCategoryRefDataset',        # ← new dataset
    data_root=data_root,
    ann_file='mdetr_annotations/single_category_ref_4_19.json',
    data_prefix=dict(img='single_category_ref/'),
    test_mode=True,
    return_classes=True,
    # optionally keep only one category
    filter_cfg=dict(filter_empty_gt=False, cat_ids=[1]),
    pipeline=test_pipeline,
)

val_evaluator_single_cat = dict(
    type='SingleCategoryRefMetric',                # or 'RefExpMetric' if you prefer the original bbox‑F1
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