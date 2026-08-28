import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from cascade.data import VideoIdentityDataset


class FakeDeepfakeBenchDataset(Dataset):
    def __init__(self):
        self.data_dict = {
            'image': ['a/0.png', 'b/0.png', 'a/1.png', 'c/0.png'],
            'label': [0, 1, 0, 1],
            'video_name': ['video_a', 'video_b', 'video_a', 'video_c'],
        }

    def __len__(self):
        return len(self.data_dict['image'])

    def __getitem__(self, index):
        return torch.tensor([index]), self.data_dict['label'][index], None, None

    @staticmethod
    def collate_fn(batch):
        images, labels, _, _ = zip(*batch)
        return {
            'image': torch.stack(images),
            'label': torch.tensor(labels),
            'landmark': None,
            'mask': None,
        }


class VideoIdentityDatasetTest(unittest.TestCase):
    def test_collate_adds_batch_aligned_video_names(self):
        dataset = VideoIdentityDataset(FakeDeepfakeBenchDataset())
        loader = DataLoader(
            dataset,
            batch_size=3,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )

        batch = next(iter(loader))

        self.assertEqual(batch['video_name'], ['video_a', 'video_b', 'video_a'])
        self.assertEqual(batch['label'].tolist(), [0, 1, 0])

    def test_video_subset_contains_all_and_only_requested_frames(self):
        dataset = VideoIdentityDataset(FakeDeepfakeBenchDataset())

        indices = dataset.indices_for_videos({'video_a', 'video_c'})
        subset = dataset.subset_for_videos({'video_a', 'video_c'})

        self.assertEqual(indices, [0, 2, 3])
        self.assertEqual(len(subset), 3)
        self.assertEqual(
            [subset[index][1] for index in range(len(subset))],
            ['video_a', 'video_a', 'video_c'],
        )

    def test_unknown_video_is_rejected(self):
        dataset = VideoIdentityDataset(FakeDeepfakeBenchDataset())

        with self.assertRaises(KeyError):
            dataset.indices_for_videos({'missing_video'})

    def test_conflicting_labels_for_one_video_are_rejected(self):
        base_dataset = FakeDeepfakeBenchDataset()
        base_dataset.data_dict['label'][2] = 1

        with self.assertRaises(ValueError):
            VideoIdentityDataset(base_dataset)


if __name__ == '__main__':
    unittest.main()
