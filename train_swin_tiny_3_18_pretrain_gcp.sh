#!/bin/bash

# Variables
n=${1:-8}
epochs=${2:-30}
exp_name=${3:-default}
pretrained_weights=${4:-"None"}
config_path=${5:-"None"}
resume_mode=${6:-resume}  # Set to "overwrite" (default) or "resume"
# Updated default output directory for experiment results:
output_dir=${7:-"/exp_outputs"}

# Install required packages
apt-get update
apt-get install libstdc++6 -y
apt-get install libgl1-mesa-glx -y
apt-get install libglib2.0-0 -y
apt-get install g++ -y

# Set output directory to save the experiment results
OUTPUT_DIR=${output_dir}/${exp_name}

# Handle resume or overwrite mode
if [ "$resume_mode" = "overwrite" ]; then
    echo "Overwrite mode: Deleting previous outputs in ${OUTPUT_DIR} (if any)."
    rm -rf ${OUTPUT_DIR}
    resume_flag=""
elif [ "$resume_mode" = "resume" ]; then
    echo "Resume mode: Training will resume from existing outputs (if available)."
    resume_flag="--resume"
else
    echo "Unknown resume_mode '${resume_mode}', defaulting to overwrite."
    rm -rf ${OUTPUT_DIR}
    resume_flag=""
fi

# Run distributed training using --cfg-options to override parameters, including load_from
./tools/dist_train.sh ${config_path} $n \
    --cfg-options train_cfg.max_epochs=$epochs work_dir=${OUTPUT_DIR} load_from=${pretrained_weights} ${resume_flag}

# Example test commands (uncomment if needed)
# ./tools/dist_test.sh ${config_path} ${pretrained_weights} ${n} --eval coco,lvis,odinw --cfg-options work_dir=${OUTPUT_DIR}
# python tools/test.py ${config_path} ${pretrained_weights} --cfg-options work_dir=${OUTPUT_DIR}