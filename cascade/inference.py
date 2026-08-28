"""Detector inference helpers shared by cascade stages."""

import torch
from tqdm import tqdm

from .video_scoring import VideoScoreAccumulator


@torch.no_grad()
def run_frame_detector_stage(model, data_loader, device, description=None):
    """Run a frame detector and return arithmetic-mean video scores."""
    model.eval()
    accumulator = VideoScoreAccumulator()

    progress = tqdm(data_loader, total=len(data_loader), desc=description)
    for data_dict in progress:
        if 'video_name' not in data_dict:
            raise KeyError('Cascade batches must contain video_name.')

        # binarify the labels
        labels = torch.where(data_dict['label'] != 0, 1, 0).long()  # CPU Tensor

        data_dict['label'] = labels.to(device)                      # GPU Tensor
        data_dict['image'] = data_dict['image'].to(device)

        for optional_key in ('mask', 'landmark'):
            optional_value = data_dict.get(optional_key)
            if torch.is_tensor(optional_value):
                data_dict[optional_key] = optional_value.to(device)

        predictions = model(data_dict, inference=True)
        # 強制 Detector 必須 return prob，我們就是要 prob
        if 'prob' not in predictions:
            raise KeyError(
                'Detector inference output must contain fake probability "prob".'
            )

        probabilities = predictions['prob'].detach().cpu().reshape(-1).tolist()
        
        accumulator.update(
            probabilities=probabilities,
            labels=labels.reshape(-1).tolist(),
            video_names=data_dict['video_name'],
        )

    return accumulator.finalize()
