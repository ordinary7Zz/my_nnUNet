#!/usr/bin/env bash
set -euo pipefail

dataset_root="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Superimposed_experiment/dataset_1/train"
images_subdir="images"
labels_subdir="masks"
output_json="./dataset.json"

python generate_dataset_json.py "$dataset_root" \
  --images-subdir "$images_subdir" \
  --labels-subdir "$labels_subdir" \
  --output-json "$output_json"
