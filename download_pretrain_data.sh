set -e  # Exit immediately if any command fails

REMOTE_BUCKET="gs://oe-training-jieyu/weikaih/grounding_dino_data"
LOCAL_DATA_DIR="/mmdetection/data"

echo "Starting concurrent downloads..."

gsutil -m cp -r "${REMOTE_BUCKET}/coco" "${LOCAL_DATA_DIR}/" &
gsutil -m cp -r "${REMOTE_BUCKET}/flickr30k_entities" "${LOCAL_DATA_DIR}/" &
gsutil -m cp -r "${REMOTE_BUCKET}/gqa" "${LOCAL_DATA_DIR}/" &
gsutil -m cp -r "${REMOTE_BUCKET}/objects365v1" "${LOCAL_DATA_DIR}/" &
gsutil -m cp -r "${REMOTE_BUCKET}/odinw" "${LOCAL_DATA_DIR}/" &

# Wait for all background processes to finish
wait

echo "All downloads completed."