#!/usr/bin/env bash
set -e

REMOTE_BUCKET="gs://oe-training-jieyu/weikaih/grounding_dino_synthetic_data/ovd_1m"
LOCAL_ROOT="/mmdetection/data/ovd"

echo "Preparing local directory…"
mkdir -p \
  "${LOCAL_ROOT}/panoptic_train" \
  "${LOCAL_ROOT}/annotations" \
  "${LOCAL_ROOT}/train"

echo "Starting selective downloads…"

# 1) panoptic_train annotations only
gsutil -m cp -r \
  "${REMOTE_BUCKET}/panoptic_train/*" \
  "${LOCAL_ROOT}/panoptic_train" &

# 2) top‑level annotation folder (singular)
gsutil -m rsync -r -x "separate_annotations/" ${REMOTE_BUCKET}/annotations ${LOCAL_ROOT}/annotations &

# 3) train_keep_color → renamed locally to 'train'
gsutil -m cp -r \
  "${REMOTE_BUCKET}/train_keep_color/*" \
  "${LOCAL_ROOT}/train" &

wait

echo "All done!"