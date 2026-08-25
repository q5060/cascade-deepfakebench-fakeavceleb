# Files
將以下檔案放入對應資料夾中
- `preprocessing/datasets_json/`:
  - `Celeb-DF-v2.json`
  - `DFDC.json`
  - `FaceForensics++.json`
- `preprocessing/dlib_tools`:`shape_predictor_81_face_landmarks.dat`
- `training/weights/`: 和林禹丞拿或到到 repo release 下載
  - `ucf_best.pth`
  - `xception_best.pth`
- `training/pretrained/`： repo release 下載
  - `xception-b5690688.pth`
  - `hrnetv2_w48_imagenet_pretrained.pth`
  - `efficientnet-b4-6ed6700e.pth`

# Docker

## build
```bash
docker build -t cascade-deepfakebench .
```
### 林禹丞 自己的電腦
```powershell
docker run --name cascade-deepfakebench `
  --gpus all `
  -itd `
  -v "C:\大學\2026暑假\專題\Cascade-DeepfakeBench:/app" `
  --shm-size 64G `
  cascade-deepfakebench
```

### 實驗室電腦：專題2024 PC 4090
```powershell
docker run --name cascade-deepfakebench `
  --gpus all `
  -itd `
  -v "C:\Users\oplab\Desktop\Cascade-DeepfakeBench:/app" `
  -v "E:\114_IMProject\data:/app/datasets/rgb" `
  --shm-size 64G `
  cascade-deepfakebench
```

## start
```bash
docker start cascade-deepfakebench
```

## 啟動後進入 container
```bash
docker exec -it cascade-deepfakebench /bin/bash
```

## 暫停使用 docker
```bash
docker stop cascade-deepfakebench
```


# 測試
```bash
python training/test.py \
--detector_path training/config/detector/xception.yaml \
--test_dataset FaceForensics++ \
--weights_path ./training/weights/xception_best.pth
```

```bash
python training/test.py \
--detector_path training/config/detector/xception.yaml \
--test_dataset Celeb-DF-v2 \
--weights_path ./training/weights/xception_best.pth
```