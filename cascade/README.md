# Cascade pipeline

New sequential-cascade components live in this directory. The original
DeepfakeBench code should only receive small compatibility changes when the
cascade cannot be implemented cleanly as an adapter.

## Video identity contract

`video_name` is an opaque, non-empty string that uniquely identifies one video
within a test dataset. Cascade code must compare the complete value and must not
derive it by splitting a frame path.

For every dataset sample, these lists must stay index-aligned:

```text
data_dict["image"][i]
data_dict["label"][i]
data_dict["video_name"][i]
```

`VideoIdentityDataset` preserves the original DeepfakeBench sample and collate
formats while adding this batch metadata（原本 Dataset 的 Wrapper）:

```python
batch["video_name"]  # list[str], aligned with batch["image"]
```

The adapter also provides `indices_for_videos()` and `subset_for_videos()` so a
later stage can receive all and only the frames belonging to uncertain videos.

## Stage 1: Xception

Inside the Docker container, run the first stage from the repository mount:

```bash
cd /app
python cascade/test.py \
  --low_threshold 0.2 \
  --high_threshold 0.8
```

Replace the example thresholds with the operational values supplied to the
system. The stage groups frames by the opaque `video_name`, computes the
arithmetic mean fake probability, and applies these inclusive exit rules:

```text
score <= low_threshold   -> REAL
score >= high_threshold  -> FAKE
otherwise                -> UNCERTAIN
```

Stage 1 AUC is calculated over every FaceForensics++ video. A cohort with fewer
than two ground-truth classes is reported as `N/A`.
