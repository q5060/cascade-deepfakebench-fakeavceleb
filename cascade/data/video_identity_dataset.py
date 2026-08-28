"""Attach stable video identity metadata to DeepfakeBench samples."""

from collections import defaultdict
from typing import Iterable

from torch.utils.data import Dataset, Subset


class VideoIdentityDataset(Dataset):
    """Adapt a DeepfakeBench dataset to emit ``video_name`` in every batch.

    The wrapped dataset keeps its original sample format and ``collate_fn``.
    ``video_name`` is treated as an opaque identifier: cascade code groups and
    routes by exact equality and must not recover identity by parsing a path.
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

        data_dict = getattr(dataset, 'data_dict', None)     # dataset.data_dict
        if not isinstance(data_dict, dict) or 'video_name' not in data_dict:
            raise ValueError(
                'The wrapped dataset must expose data_dict["video_name"].'
            )

        # 確認 video_name_list[i] 必須永遠和 dataset[i] 是同一個 sample
        self.video_name_list = list(data_dict['video_name'])
        # consistency check: video_name_list must have the same length as dataset
        if len(self.video_name_list) != len(dataset):
            raise ValueError(
                'The number of video names must equal the number of samples.'
            )
        
        """
        # record samples for each video identity as such:
        
        self._video_to_indices = {
            "video_A": [1, 3, 9, 20],
            "video_B": [0, 4, 7],
            "video_C": [2, 5, 6],
        }
        """
        self._video_to_indices = defaultdict(list)
        for index, video_name in enumerate(self.video_name_list):
            if not isinstance(video_name, str) or not video_name:
                raise ValueError(
                    f'video_name at index {index} must be a non-empty string.'
                )
            self._video_to_indices[video_name].append(index)
        # check if all samples of a video have the same label, if not throw exception
        self._validate_video_labels(data_dict.get('label'))

    def _validate_video_labels(self, labels):
        """Reject a video identity that is associated with multiple labels."""
        if labels is None:
            return
        if len(labels) != len(self.video_name_list):
            raise ValueError(
                'The number of labels must equal the number of video names.'
            )

        video_labels = {}
        for video_name, label in zip(self.video_name_list, labels):
            if hasattr(label, 'item'):
                label = label.item()
            previous = video_labels.setdefault(video_name, label)
            if previous != label:
                raise ValueError(
                    f'Video {video_name!r} is associated with multiple labels.'
                )

    @property
    def video_names(self):
        """Return all distinct video identities in their first-seen order."""
        return tuple(self._video_to_indices.keys())

    def indices_for_videos(self, video_names: Iterable[str]):
        """Return sample indices for the requested videos in dataset order."""
        requested = set(video_names)
        unknown = requested.difference(self._video_to_indices)      # 有在 requested 但沒有在原本影片裡，代表有問題
        if unknown:
            unknown_text = ', '.join(sorted(unknown))
            raise KeyError(f'Unknown video_name values: {unknown_text}')

        return [
            index
            for index, video_name in enumerate(self.video_name_list)
            if video_name in requested
        ]

    def subset_for_videos(self, video_names: Iterable[str]):
        """Build a subset containing all and only the requested videos."""
        return Subset(self, self.indices_for_videos(video_names))

    def __getitem__(self, index):
        return self.dataset[index], self.video_name_list[index]

    def __len__(self):
        return len(self.dataset)

    def collate_fn(self, batch):
        """Use the original collator, then attach batch-aligned video names."""
        samples, video_names = zip(*batch)
        base_collate_fn = getattr(self.dataset, 'collate_fn', None)     # 使用 DeepfakeAbstractBaseDataset.collate_fn
        if base_collate_fn is None:
            raise ValueError('The wrapped dataset must provide collate_fn.')

        data_dict = base_collate_fn(list(samples))
        if not isinstance(data_dict, dict):
            raise TypeError('The wrapped dataset collate_fn must return a dict.')

        data_dict['video_name'] = list(video_names)
        return data_dict
