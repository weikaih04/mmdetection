# n=${1:-2}
# stage2_iter=${2:-10000}
# exp_name=${3:-default}
# config_file=${4:-configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep_default.yaml}

# # setup keys
# # export COCO_10K="/input/jieyuz2/weikaih/data/mask2former_dataset/datasets/coco_10k"
# # export COCO_50K="/input/jieyuz2/weikaih/data/mask2former_dataset/datasets/coco_50k"
# # export SYNTHETIC_DATA_PATH="$synthetic_data_path"
# # export EXP_NAME="$exp_name"
# # export DETECTRON2_DATASETS=/input/jieyuz2/weikaih/data/mask2former_dataset/datasets
# # export DATASETS=/input/jieyuz2/weikaih/data/mask2former_dataset/datasets
# # export WANDB_API_KEY=f773908953fc7bea7008ae1cf3701284de1a0682

# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

cd /input_oe_data/jieyuz/weikaih/code/mmdetection
# mkdir -p /input_oe_data/jieyuz/weikaih/code/mmdetection/data
# for item in /input_oe_data/jieyuz/weikaih/grounding_dino_data/*; do
#     ln -s "$item" /input_oe_data/jieyuz/weikaih/code/mmdetection/data/
# done

apt-get update
apt-get install libstdc++6 -y
apt-get install libgl1-mesa-glx -y
apt-get install libglib2.0-0 -y
apt-get install g++ -y

# python tools/train.py configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_debug.py --amp \
# ./tools/dist_train.sh configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_debug.py 4
# ./tools/dist_train.sh configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_debug.py  4
./tools/dist_train.sh configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_debug.py 8


