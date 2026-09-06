"""Run FakeAVCeleb preprocess, JSON, and score CSV export. Use inside Docker."""

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESS_DIR = PROJECT_ROOT / 'preprocessing'
OUTPUT_DIR = PROJECT_ROOT / 'output'


def run(command, cwd):
    print('>>>', ' '.join(command), flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print('=== 1/4 Crop faces ===', flush=True)
    run(
        [sys.executable, 'preprocess.py', '--dataset_name', 'FakeAVCeleb'],
        PREPROCESS_DIR,
    )

    print('=== 2/4 Write FakeAVCeleb.json ===', flush=True)
    run(
        [
            sys.executable,
            'rearrange.py',
            '--dataset_name',
            'FakeAVCeleb',
            '--output_file_path',
            './dataset_json',
        ],
        PREPROCESS_DIR,
    )

    print('=== 3/4 Score with Xception ===', flush=True)
    run(
        [
            sys.executable,
            'scripts/export_score_csv.py',
            '--detector_path',
            'training/config/detector/xception.yaml',
            '--weights_path',
            'training/weights/xception_best.pth',
            '--test_dataset',
            'FakeAVCeleb',
            '--output_csv',
            'output/scores_visual_xception_fakeavceleb_v0.csv',
        ],
        PROJECT_ROOT,
    )

    print('=== 4/4 Score with UCF ===', flush=True)
    run(
        [
            sys.executable,
            'scripts/export_score_csv.py',
            '--detector_path',
            'training/config/detector/ucf.yaml',
            '--weights_path',
            'training/weights/ucf_best.pth',
            '--test_dataset',
            'FakeAVCeleb',
            '--output_csv',
            'output/scores_visual_ucf_fakeavceleb_v0.csv',
        ],
        PROJECT_ROOT,
    )

    print('Done. CSV files are in /app/output/', flush=True)
    for path in sorted(OUTPUT_DIR.glob('*.csv')):
        print(path, flush=True)


if __name__ == '__main__':
    main()
