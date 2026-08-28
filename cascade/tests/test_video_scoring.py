import unittest

from cascade.metrics import operational_auc
from cascade.video_scoring import (
    FAKE,
    REAL,
    UNCERTAIN,
    VideoScore,
    VideoScoreAccumulator,
    route_video_scores,
)


class VideoScoreAccumulatorTest(unittest.TestCase):
    def test_arithmetic_mean_does_not_depend_on_frame_order(self):
        accumulator = VideoScoreAccumulator()
        accumulator.update(
            probabilities=[0.2, 0.9, 0.4, 0.7],
            labels=[0, 1, 0, 1],
            video_names=['video_a', 'video_b', 'video_a', 'video_b'],
        )

        scores = accumulator.finalize()

        self.assertEqual([score.video_name for score in scores], ['video_a', 'video_b'])
        self.assertAlmostEqual(scores[0].score, 0.3)
        self.assertAlmostEqual(scores[1].score, 0.8)
        self.assertEqual(scores[0].frame_count, 2)

    def test_conflicting_frame_labels_are_rejected(self):
        accumulator = VideoScoreAccumulator()
        accumulator.update([0.1], [0], ['video_a'])

        with self.assertRaises(ValueError):
            accumulator.update([0.2], [1], ['video_a'])


class DoubleThresholdRoutingTest(unittest.TestCase):
    def test_threshold_boundaries_are_inclusive_exits(self):
        scores = (
            VideoScore('low', 0, 0.2, 1),
            VideoScore('middle', 1, 0.5, 1),
            VideoScore('high', 1, 0.8, 1),
        )

        routed = route_video_scores(scores, 0.2, 0.8)

        self.assertEqual(
            [video.decision for video in routed],
            [REAL, UNCERTAIN, FAKE],
        )

    def test_invalid_threshold_order_is_rejected(self):
        with self.assertRaises(ValueError):
            route_video_scores([], 0.8, 0.2)


class OperationalAucTest(unittest.TestCase):
    def test_auc_uses_only_supplied_stage_cohort(self):
        scores = (
            VideoScore('real', 0, 0.1, 2),
            VideoScore('fake', 1, 0.9, 2),
        )

        self.assertEqual(operational_auc(scores), 1.0)

    def test_auc_is_na_for_single_class(self):
        scores = (VideoScore('real', 0, 0.1, 2),)

        self.assertIsNone(operational_auc(scores))


if __name__ == '__main__':
    unittest.main()
