# build
```bash
docker build -t cascade-deepfakebench .
```
```bash
docker run --name cascade-deepfakebench `
  --gpus all `
  -itd `
  -v "C:\大學\2026暑假\專題\Cascade-DeepfakeBench:/app" `
  --shm-size 64G `
  cascade-deepfakebench
```

# start
```bash
docker start cascade-deepfakebench
```

# 啟動後進入 container
```bash
docker exec -it cascade-deepfakebench /bin/bash
```

# 暫停使用 docker
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
--test_dataset FaceForensics++ Celeb-DF-v2 DFDC \
--weights_path ./training/weights/xception_best.pth
```