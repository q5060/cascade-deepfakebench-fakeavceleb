"""Export per-video detector scores in the team's scores_visual CSV format."""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / 'training'
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from cascade.data import VideoIdentityDataset
from cascade.test import (
    build_model,
    choose_device,
    initialize_seed,
    load_config,
    project_path,
)
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset


CSV_COLUMNS = [
    'sample_id',
    'dataset',
    'label',
    'detector_name',
    'modality',
    'fake_score',
    'score_type',
    'inference_time_ms',
    'window_start_sec',
    'window_end_sec',
    'status',
    'error_message',
]

DETECTOR_DISPLAY_NAMES = {
    'ucf': 'UCF',
    'xception': 'Xception',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Write per-video fake scores to a scores_visual CSV.'
    )
    parser.add_argument('--detector_path', required=True)
    parser.add_argument('--weights_path', required=True)
    parser.add_argument('--test_dataset', default='FakeAVCeleb')
    parser.add_argument('--output_csv', required=True)
    parser.add_argument(
        '--test_config',
        default='training/config/test_config.yaml',
    )
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--device')
    return parser.parse_args()


def detector_display_name(model_name):
    return DETECTOR_DISPLAY_NAMES.get(model_name, model_name)


def collect_video_scores(model, data_loader, device):
    model.eval()
    score_sums = defaultdict(float)
    frame_counts = defaultdict(int)
    labels = {}
    elapsed_ms = defaultdict(float)

    progress = tqdm(data_loader, total=len(data_loader), desc='Scoring videos')
    with torch.no_grad():
        for data_dict in progress:
            started = time.perf_counter()
            batch_labels = torch.where(data_dict['label'] != 0, 1, 0).long()
            data_dict['label'] = batch_labels.to(device)
            data_dict['image'] = data_dict['image'].to(device)
            for optional_key in ('mask', 'landmark'):
                optional_value = data_dict.get(optional_key)
                if torch.is_tensor(optional_value):
                    data_dict[optional_key] = optional_value.to(device)

            predictions = model(data_dict, inference=True)
            probabilities = predictions['prob'].detach().cpu().reshape(-1).tolist()
            video_names = list(data_dict['video_name'])
            label_values = batch_labels.reshape(-1).tolist()
            batch_ms = (time.perf_counter() - started) * 1000.0
            share_ms = batch_ms / max(len(video_names), 1)

            for probability, label, video_name in zip(
                probabilities, label_values, video_names
            ):
                score_sums[video_name] += float(probability)
                frame_counts[video_name] += 1
                elapsed_ms[video_name] += share_ms
                previous = labels.setdefault(video_name, int(label))
                if previous != int(label):
                    raise ValueError(
                        'Video {!r} has more than one label.'.format(video_name)
                    )

    rows = []
    for video_name in score_sums:
        rows.append(
            {
                'video_name': video_name,
                'label': labels[video_name],
                'score': score_sums[video_name] / frame_counts[video_name],
                'inference_time_ms': elapsed_ms[video_name],
            }
        )
    return rows


def write_csv(rows, output_csv, dataset_name, detector_name):
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    'sample_id': row['video_name'],
                    'dataset': dataset_name,
                    'label': row['label'],
                    'detector_name': detector_name,
                    'modality': 'visual',
                    'fake_score': '{:.4f}'.format(row['score']),
                    'score_type': 'probability',
                    'inference_time_ms': '{:.2f}'.format(row['inference_time_ms']),
                    'window_start_sec': 'N/A',
                    'window_end_sec': 'N/A',
                    'status': 'ok',
                    'error_message': '',
                }
            )
    return output_path


def main():
    args = parse_args()
    os.chdir(str(PROJECT_ROOT))
    config = load_config(
        detector_path=args.detector_path,
        test_config_path=args.test_config,
        test_dataset=args.test_dataset,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    initialize_seed(config)
    device = choose_device(args.device)
    _, data_loader = _build_loader(config, device)
    model = build_model(config, project_path(args.weights_path), device)
    rows = collect_video_scores(model, data_loader, device)
    output_path = write_csv(
        rows,
        args.output_csv,
        dataset_name=args.test_dataset,
        detector_name=detector_display_name(config['model_name']),
    )
    print('Wrote {} rows to {}'.format(len(rows), output_path))


def _build_loader(config, device):
    base_dataset = DeepfakeAbstractBaseDataset(config=config, mode='test')
    dataset = VideoIdentityDataset(base_dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config['test_batchSize'],
        shuffle=False,
        num_workers=int(config['workers']),
        collate_fn=dataset.collate_fn,
        drop_last=False,
        pin_memory=device.type == 'cuda',
    )
    return dataset, data_loader


if __name__ == '__main__':
    main()
