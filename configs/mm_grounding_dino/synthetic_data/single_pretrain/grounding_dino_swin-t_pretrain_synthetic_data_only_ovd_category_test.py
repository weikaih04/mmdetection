_base_ = 'grounding_dino_swin-t_pretrain_synthetic_data_only_base.py'

train_dataloader = dict(
    dataset=dict(datasets=[ovd_phrase_dataset]))