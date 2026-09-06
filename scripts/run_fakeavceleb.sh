#!/bin/bash
# Run inside Docker from /app. Does not edit config.yaml.
set -e
cd /app

echo "=== 1/4 Crop faces ==="
cd /app/preprocessing
python preprocess.py --dataset_name FakeAVCeleb

echo "=== 2/4 Write FakeAVCeleb.json ==="
python rearrange.py --dataset_name FakeAVCeleb --output_file_path ./dataset_json

echo "=== 3/4 Score with Xception ==="
cd /app
mkdir -p output
python scripts/export_score_csv.py \
  --detector_path training/config/detector/xception.yaml \
  --weights_path training/weights/xception_best.pth \
  --test_dataset FakeAVCeleb \
  --output_csv output/scores_visual_xception_fakeavceleb_v0.csv

echo "=== 4/4 Score with UCF ==="
python scripts/export_score_csv.py \
  --detector_path training/config/detector/ucf.yaml \
  --weights_path training/weights/ucf_best.pth \
  --test_dataset FakeAVCeleb \
  --output_csv output/scores_visual_ucf_fakeavceleb_v0.csv

echo "Done. CSV files are in /app/output/"
ls -l /app/output
