"""Run the first Xception stage of the sequential cascade.

Docker example:

python cascade/test.py \
    --low_threshold 0.2 \
    --high_threshold 0.8
"""

import argparse
import os
import random
import sys
from collections import Counter
from pathlib import Path

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / 'training'
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from cascade.data import VideoIdentityDataset
from cascade.inference import run_frame_detector_stage
from cascade.metrics import operational_auc
from cascade.video_scoring import route_video_scores, validate_double_threshold
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from detectors import DETECTOR


def project_path(value):
    """Resolve command-line paths consistently inside the Docker mount."""
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run Xception video aggregation and double-threshold routing.'
    )
    parser.add_argument(
        '--detector_path',
        default='training/config/detector/xception.yaml',
    )
    parser.add_argument(
        '--test_config',
        default='training/config/test_config.yaml',
    )
    parser.add_argument(
        '--weights_path',
        default='training/weights/xception_best.pth',
    )
    parser.add_argument('--test_dataset', default='FaceForensics++')
    parser.add_argument('--low_threshold', type=float, required=True)
    parser.add_argument('--high_threshold', type=float, required=True)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--device')
    return parser.parse_args()


def load_config(args):
    with project_path(args.detector_path).open('r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    with project_path(args.test_config).open('r', encoding='utf-8') as file:
        config.update(yaml.safe_load(file))

    config['test_dataset'] = args.test_dataset      # cmd line overwrite config
    if args.batch_size is not None:
        config['test_batchSize'] = args.batch_size
    if args.workers is not None:
        config['workers'] = args.workers

    config['pretrained'] = str(project_path(config['pretrained']))
    config['dataset_json_folder'] = str(
        project_path(config['dataset_json_folder'])
    )
    config['rgb_dir'] = str(project_path(config['rgb_dir']))
    config['lmdb_dir'] = str(project_path(config['lmdb_dir']))
    return config


def initialize_seed(config):
    seed = config.get('manualSeed')
    if seed is None:
        seed = 1024
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_argument):
    if device_argument:
        device = torch.device(device_argument)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available in Docker.')
    return device


def build_data_loader(config, device):
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


def build_model(config, weights_path, device):
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    checkpoint = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    return model


def print_stage_summary(config, video_scores, routed_videos, args):
    counts = Counter(video.decision for video in routed_videos)
    auc = operational_auc(video_scores)
    frame_count = sum(video.frame_count for video in video_scores)

    print('\n=== Cascade Stage 1: {} ==='.format(config['model_name']))
    print('dataset: {}'.format(config['test_dataset']))
    print('frames: {}'.format(frame_count))
    print('videos: {}'.format(len(video_scores)))
    print('low_threshold: {}'.format(args.low_threshold))
    print('high_threshold: {}'.format(args.high_threshold))
    print('REAL: {}'.format(counts['REAL']))
    print('UNCERTAIN: {}'.format(counts['UNCERTAIN']))
    print('FAKE: {}'.format(counts['FAKE']))
    print('AUC: {}'.format('N/A' if auc is None else '{:.6f}'.format(auc)))


def main():
    args = parse_args()
    validate_double_threshold(args.low_threshold, args.high_threshold)
    os.chdir(str(PROJECT_ROOT))
    config = load_config(args)
    initialize_seed(config)
    device = choose_device(args.device)

    _, data_loader = build_data_loader(config, device)
    model = build_model(config, project_path(args.weights_path), device)
    video_scores = run_frame_detector_stage(
        model=model,
        data_loader=data_loader,
        device=device,
        description='Stage 1: Xception',
    )
    routed_videos = route_video_scores(
        video_scores,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
    )
    print_stage_summary(config, video_scores, routed_videos, args)

if __name__ == '__main__':
    main()
