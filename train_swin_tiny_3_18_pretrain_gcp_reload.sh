#!/bin/bash

# -----------------------------------------------------------
#
# Usage:
#   bash train_and_retry.sh <num_gpus> <epochs> <exp_name> <pretrained_weights> \
#                          <config_path> <resume_mode> <output_dir> <gcs_bucket_path>
#
#   resume_mode: "overwrite" (clear work_dir) or "resume" (pull+keep checkpoints + --auto-resume)
# -----------------------------------------------------------

# === Arguments ===
n=${1:-8}
epochs=${2:-30}
exp_name=${3:-default_exp}
pretrained_weights=${4:-"None"}      # e.g. /path/to/pretrained.pth
config_path=${5:-"None"}             # e.g. configs/xxx.py
resume_mode=${6:-resume}             # "overwrite" or "resume"
output_root=${7:-"/exp_outputs"}     # base dir for all experiments
gcs_bucket=${8:-"gs://your-bucket/your-path"}  # root bucket path

MAX_RETRY=10

if [ "${config_path}" = "None" ]; then
  echo "Error: please provide a config path as argument 5."
  exit 1
fi

# === Prepare work_dir ===
OUTPUT_DIR=${output_root}/${exp_name}

if [ "${resume_mode}" = "overwrite" ]; then
  echo "[MODE] overwrite → clearing ${OUTPUT_DIR}"
  rm -rf "${OUTPUT_DIR}"
elif [ "${resume_mode}" = "resume" ]; then
  echo "[MODE] resume → creating and populating ${OUTPUT_DIR} from GCS"
  mkdir -p "${OUTPUT_DIR}"
  # Pull down existing outputs if any
  gsutil -m cp -r "${gcs_bucket}/${exp_name}/"* "${OUTPUT_DIR}/" || \
    echo "⚠️ No existing remote outputs to pull down."
else
  echo "[WARN] Unknown resume_mode '${resume_mode}', defaulting to overwrite"
  rm -rf "${OUTPUT_DIR}"
fi

mkdir -p "${OUTPUT_DIR}"

# === Install deps if needed ===
apt-get update && \
apt-get install -y libstdc++6 libgl1-mesa-glx libglib2.0-0 g++ && \
source /etc/profile

# === Build common cfg-options ===
CFG_OPTS="train_cfg.max_epochs=${epochs} work_dir=${OUTPUT_DIR}"
if [ "${pretrained_weights}" != "None" ]; then
  CFG_OPTS="${CFG_OPTS} load_from=${pretrained_weights}"
fi

# === Resume flag ===
if [ "${resume_mode}" = "resume" ]; then
  RESUME_FLAG="--resume"
else
  RESUME_FLAG=""
fi

# === Retry Loop with GCS Copy ===
attempt=0
while true; do
  echo
  echo "=== Attempt $((attempt+1)) starting at $(date) ==="
  ./tools/dist_train.sh "${config_path}" "${n}" \
    --cfg-options ${CFG_OPTS} \
    ${RESUME_FLAG}
  exit_code=$?

  if [ $exit_code -eq 0 ]; then
    echo "✅ Training succeeded on attempt $((attempt+1))."
    break
  fi

  attempt=$((attempt+1))
  echo "⚠️ Training crashed (exit ${exit_code}) on attempt ${attempt}."

  # Copy partial results back to GCS
  echo "⏫ Copying partial outputs to ${gcs_bucket}/${exp_name}/ …"
  gsutil -m cp -r "${OUTPUT_DIR}" "${gcs_bucket}/${exp_name}/"

  if [ $attempt -ge $MAX_RETRY ]; then
    echo "❌ Reached max retries (${MAX_RETRY}). Exiting."
    exit $exit_code
  fi

  echo "🔁 Retrying in 5s…"
  sleep 5
done

# === Final copy after success ===
echo "⏫ Final copy of outputs to ${gcs_bucket}/${exp_name}/ …"
gsutil -m cp -r "${OUTPUT_DIR}" "${gcs_bucket}/${exp_name}/"

echo "All done. Outputs in ${OUTPUT_DIR} and at ${gcs_bucket}/${exp_name}/"