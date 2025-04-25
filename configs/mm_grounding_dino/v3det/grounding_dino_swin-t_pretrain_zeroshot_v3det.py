_base_ = '../grounding_dino_swin-t_pretrain_obj365.py'

model = dict(test_cfg=dict(
    max_per_img=300,
    chunked_size=40,
))

data_root = 'data/v3det/'

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/v3det_2023_v1_train.json',
        # label_map_file='annotations/v3det_2023_v1_label_map.json',
        data_prefix=dict(img='')))
test_dataloader = val_dataloader

# numpy < 1.24.0
val_evaluator = dict(
    # _delete_=True,
    # type='COCOMetric',
    ann_file=data_root + 'annotations/v3det_2023_v1_train.json')
test_evaluator = val_evaluator
