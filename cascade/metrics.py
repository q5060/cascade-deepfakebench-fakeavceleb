"""Metrics for operational cascade cohorts."""

from sklearn.metrics import roc_auc_score


def operational_auc(video_scores):
    """Compute AUC for exactly the videos processed by one stage.

    ``None`` represents N/A when the stage has no videos or only one class.
    """
    labels = [video.label for video in video_scores]
    if len(set(labels)) < 2:
        return None
    scores = [video.score for video in video_scores]
    return float(roc_auc_score(labels, scores))
