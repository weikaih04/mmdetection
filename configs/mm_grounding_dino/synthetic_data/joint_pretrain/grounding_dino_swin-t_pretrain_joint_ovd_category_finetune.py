_base_ = 'grounding_dino_swin-t_pretrain_joint_base.py'

max_epochs = 10
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[6],
        gamma=0.1)
]
train_cfg = dict(max_epochs=max_epochs, val_interval=1)