_base_ = '../../grounding_dino_swin-t_pretrain_base.py'

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0004,
                   weight_decay=0.0001),  # bs=16 0.0001
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0.1),
        }
    ),
)



# ##### evaluation setup, coco, lvis and odinw

# # --- Define each evaluation dataset & evaluator ---
# # (Your COCO evaluation settings are assumed to be defined already)
# # coco_dataset = dict(
# #     type='CocoDataset',
# #     ann_file='annotations/instances_val2017.json',
# #     data_prefix=dict(img='val2017/'),
# #     pipeline=test_pipeline
# # )
# # coco_evaluator = dict(
# #     type='CocoMetric',
# #     ann_file='annotations/instances_val2017.json'
# # )


# lvis_dataset = dict(
#     type='LVISV1Dataset',
#     data_root='data/coco/',
#     ann_file='annotations/lvis_v1_minival_inserted_image_name.json',
#     data_prefix=dict(img=''),
#     pipeline=test_pipeline
# )
# lvis_evaluator = dict(
#     _delete_=True,
#     type='LVISFixedAPMetric',
#     ann_file='data/coco/annotations/lvis_v1_minival_inserted_image_name.json'
# )

# odinw_dataset = dict(
#     type='ODVGDataset',  # or the appropriate type for your OdinW data
#     data_root='data/odinw/',
#     ann_file='annotations/odinw_test.json',  # adjust as needed
#     data_prefix=dict(img=''),
#     pipeline=test_pipeline
# )
# odinw_evaluator = dict(
#     type='CocoMetric',  # or another evaluator if you use a custom one
#     ann_file='data/odinw/annotations/odinw_test.json'
# )

# # --- Combine evaluation configurations into lists ---

# val_dataloader = [
#     dict(dataset=coco_dataset),
#     dict(dataset=lvis_dataset),
#     dict(dataset=odinw_dataset)
# ]
# test_dataloader = val_dataloader

# val_evaluator = [coco_evaluator, lvis_evaluator, odinw_evaluator]
# test_evaluator = val_evaluator
model = dict(test_cfg=dict(
    max_per_img=300,
    chunked_size=40,
))

dataset_type = 'LVISV1Dataset'
data_root = 'data/coco/'

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        type=dataset_type,
        ann_file='annotations/lvis_v1_minival_inserted_image_name.json',
        data_prefix=dict(img='')))
test_dataloader = val_dataloader

# numpy < 1.24.0
val_evaluator = dict(
    _delete_=True,
    type='LVISFixedAPMetric',
    ann_file=data_root +
    'annotations/lvis_v1_minival_inserted_image_name.json')
test_evaluator = val_evaluator