# Milestone
1. 定義正式的 video identity 資料契約
   每個 sample 至少帶有 video_name，最好能保證跨 dataset 唯一，例如包含 dataset、label/type、原始 video name。
2. 完成 Xception 單 detector video routing
   收集 (video_name, label, frame_prob)，依 video 聚合，再套 low/high threshold。
3. 驗證 routing subset
   用 uncertain video IDs 找 Dataset indices，建立 Subset，確認它包含「全部且只有」uncertain videos 的 frames。
4. 接上 UCF
   統一 UCF inference output，再載入第二組 config 與 checkpoint。
5. 抽象化 Controller
   Detector 保持只處理 batch；routing、aggregation、threshold、stage state 都放在 detector 外面。建議保留 test.py 作為單模型 baseline，不把 cascade 邏輯硬塞進去。
6. Calibration 與 Evaluation
   加入 exit rate、平均 detectors/video、FAR/FRR、latency，以及最終分類結果。