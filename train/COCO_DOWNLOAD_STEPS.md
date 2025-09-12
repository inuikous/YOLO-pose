# フル COCO (Person Keypoints) ダウンロード & 学習フロー

## 0. 前提
- 空き容量: 30GB 以上推奨
- 回線が不安定なら一つずつ取得
- 作業ルート例: `C:\datasets\coco`

## 1. ダウンロード (Windows cmd / PowerShell)
```
mkdir C:\datasets\coco
cd /d C:\datasets\coco
powershell -Command "Invoke-WebRequest -Uri http://images.cocodataset.org/zips/train2017.zip -OutFile train2017.zip"
powershell -Command "Invoke-WebRequest -Uri http://images.cocodataset.org/zips/val2017.zip -OutFile val2017.zip"
powershell -Command "Invoke-WebRequest -Uri http://images.cocodataset.org/annotations/annotations_trainval2017.zip -OutFile annotations_trainval2017.zip"
```
(回線不安定なら 1 行ずつ実行)

## 2. 展開
```
powershell -Command "Expand-Archive train2017.zip -DestinationPath ."
powershell -Command "Expand-Archive val2017.zip -DestinationPath ."
powershell -Command "Expand-Archive annotations_trainval2017.zip -DestinationPath ."
```
確認:
```
train2017\ (≈118k 画像)
val2017\
annotations\person_keypoints_train2017.json
```

## 3. 小規模サブセット抽出 (高速学習用)
例: train 500 / val 100 枚
```
python prepare_coco_pose_subset.py ^
  --train-ann C:/datasets/coco/annotations/person_keypoints_train2017.json ^
  --val-ann C:/datasets/coco/annotations/person_keypoints_val2017.json ^
  --train-images C:/datasets/coco/train2017 ^
  --val-images C:/datasets/coco/val2017 ^
  --out C:/datasets/coco_person_subset ^
  --train-count 500 --val-count 100 --copy-images
```
出力: `C:/datasets/coco_person_subset/pose.yaml`

## 4. 学習 (サブセット)
```
python train_pose.py --pretrained yolo11n-pose.pt --data C:/datasets/coco_person_subset/pose.yaml --epochs 20 --batch 32 --imgsz 640
```
結果: `runs/pose/train/weights/best.pt`

## 5. OpenVINO 変換
```
python export_openvino.py --weights runs/pose/train/weights/best.pt --imgsz 640 --half
```
出力例: `runs/pose/train/weights/best_openvino_model/`

## 6. 推論テスト (画像 or Webカメラ)
```
python infer_openvino_pose.py --model runs/pose/train/weights/best_openvino_model --source 0 --show
```

## 7. フル COCO で本格学習 (時間長い)
```
yolo task=pose mode=train model=yolo11n-pose.pt data=coco-pose.yaml epochs=100 batch=64 imgsz=640
```
`coco-pose.yaml` が無い場合は Ultralytics インストール パッケージ内を参照 (`pip show ultralytics` で location 確認)。

## 8. トラブルシュート短表
| 症状 | 対応 |
|------|------|
| 404/接続失敗 | 再実行。社内プロキシなら PowerShell `-Proxy` 指定検討 |
| ZIP 展開エラー | 再ダウンロード。部分破損の可能性 |
| メモリ不足 | `--batch` を小さく / 画像数減らす |
| 学習極端に遅い (CPU) | GPU CUDA 環境を使用 / imgsz を 512 へ縮小 |

## 9. 次の推奨ステップ
- サブセットでパイプライン検証 → OK なら枚数増やす
- 精度比較: best.pt vs OpenVINO 出力
- INT8 量子化が必要なら calibration 用サブセット作成

---
完了したら: サブセット生成コマンド実行→学習→変換 の実行指示を与えてください。
