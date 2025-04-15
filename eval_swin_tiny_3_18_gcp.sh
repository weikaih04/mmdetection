#!/bin/bash
# -----------------------------------------------------------
# Evaluation Script for Multiple Configurations
#
# This script evaluates the model’s performance for the following
# config files:
#
#   1. /input_oe_data/jieyuz/weikaih/code/mmdetection/configs/mm_grounding_dino/lvis/grounding_dino_swin-t_pretrain_zeroshot_mini-lvis.py
#   2. /input_oe_data/jieyuz/weikaih/code/mmdetection/configs/mm_grounding_dino/odinw/grounding_dino_swin-t_pretrain_odinw35.py
#   3. /input_oe_data/jieyuz/weikaih/code/mmdetection/configs/mm_grounding_dino/refcoco/grounding_dino_swin-t_pretrain_zeroshot_refexp.py
#
# The script will run the evaluation for each config using the same checkpoint,
# and save the evaluation outputs (logs, result files, etc.) into separate
# subdirectories under the specified results directory.
#
# Usage:
#   bash eval_models.sh <num_gpus> <exp_name> <checkpoint> <results_dir>
#
# Example:
#   bash eval_models.sh 8 "my_eval" /path/to/checkpoint.pth /input_oe_data/jieyuz/weikaih/code/mm_grounding_dino_results/eval_results
#
# Note:
#   This example uses the single-GPU test script (tools/test.py). If you need
#   distributed testing, you can swap out the call for the distributed launcher.
# -----------------------------------------------------------

# Variables with default values if not provided
n=${1:-8}
exp_name=${2:-default_eval}
checkpoint=${3:-"None"}
results_dir=${4:-"/exp_outputs"}

# Check that checkpoint is provided
if [ "${checkpoint}" = "None" ]; then
    echo "Error: Please provide a valid checkpoint file as argument 3."
    exit 1
fi

# Define the list of config file paths
CONFIGS=(
  "configs/mm_grounding_dino/refcoco/grounding_dino_swin-t_pretrain_zeroshot_refexp_mini.py"
  "configs/mm_grounding_dino/odinw/grounding_dino_swin-t_pretrain_odinw13.py"
)

# (Optional) Install required packages if not already present
apt-get update
apt-get install libstdc++6 -y
apt-get install libgl1-mesa-glx -y
apt-get install libglib2.0-0 -y
apt-get install g++ -y

# Loop over each configuration file and run evaluation
for config_path in "${CONFIGS[@]}"; do
    # Get the base name of the config file (without extension) to use as a subfolder name.
    base=$(basename "${config_path}" .py)
    out_dir="${results_dir}/${exp_name}/${base}"
    
    # Make sure the output directory exists
    mkdir -p "${out_dir}"
    
    echo "-----------------------------------------------------------"
    echo "Evaluating config: ${config_path}"
    echo "Using checkpoint: ${checkpoint}"
    echo "Saving results to: ${out_dir}"
    echo "-----------------------------------------------------------"
    
    # Check if the config file name contains 'refexp_mini'
    if [[ "${config_path}" == *"refexp_mini"* ]]; then
        # Run evaluation using the single-GPU test script for refexp_mini
        python tools/test.py "${config_path}" "${checkpoint}" --cfg-options work_dir="${out_dir}"
    else
        # Run evaluation using distributed testing for other configs
        ./tools/dist_test.sh "${config_path}" "${checkpoint}" ${n} --cfg-options work_dir="${out_dir}"
    fi
    
    echo "Evaluation for ${base} completed."
done

echo "All evaluations completed. Check results under: ${results_dir}/${exp_name}"