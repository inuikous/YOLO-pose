# YOLO Pose 学習→OpenVINO 変換→推論 簡易手順メモ

対象スクリプト:
- `train_pose.py` : 事前学習 `yolo11n-pose.pt` などを使った再学習
- `export_openvino.py` : 学習済み `.pt` を OpenVINO IR (XML+BIN) へエクスポート
- `infer_openvino_pose.py` : OpenVINO モデルで画像/動画/カメラ推論 + 骨格描画

---
## 1. 前提
- Python 3.11 など (既存 wheels に合わせる)
- `ultralytics` インストール済み (オフライン: `pip install --no-index --find-links=./wheels ultralytics`)
- GPU で学習する場合: 対応する CUDA 版 PyTorch を事前インストール
- OpenVINO 推論: `pip install openvino>=2024.3` (オフラインなら wheel を別途用意)

---
## 2. データセット構造 (再学習時)
```
<DATASET_ROOT>/
  images/
    train/  *.jpg *.png ...
    val/    *.jpg *.png ...
  labels/
    train/  *.txt (画像と同名)
    val/    *.txt
```
YOLO Pose ラベルフォーマット (1行/1インスタンス):
```
<class_id> <bbox_cx> <bbox_cy> <bbox_w> <bbox_h> <kpt1_x> <kpt1_y> <kpt1_v> <kpt2_x> <kpt2_y> <kpt2_v> ...
```
- すべて 0〜1 正規化 (画像幅/高さで割る)
- `v` (visibility): 0=存在せず, 1=ラベル付けしたが不可視, 2=可視 (ultralytics は 0 でも計算上扱えるが慣例)
- デフォルトは 17 Keypoints (COCO). 数変更する場合: `train_pose.py --keypoints N` と skeleton 定義変更が必要

---
## 3. 学習 (Training)
### 3.1 pose.yaml 自動生成あり
```
python train_pose.py ^
  --pretrained yolo11n-pose.pt ^
  --dataset-root C:\data\pose ^
  --epochs 50 --imgsz 640 --batch 16
```
実行時 `C:\data\pose/pose.yaml` が無ければ自動生成。

### 3.2 既存 YAML を使う
```
python train_pose.py --pretrained yolo11n-pose.pt --data C:\data\pose\my_pose.yaml --epochs 50
```

### 3.3 主なオプション
| オプション | 説明 |
|------------|------|
| `--lr0` | 初期学習率 (未指定はデフォルト) |
| `--device` | 例: `0` / `0,1` / `cpu` 空なら自動 |
| `--exist-ok` | 同名 run フォルダ上書き許可 |
| `--workers` | dataloader workers |

成果物: `runs/pose/train/weights/best.pt` (ベスト), `last.pt` (最終)。

---
## 4. OpenVINO 変換 (Export)
```
python export_openvino.py --weights runs/pose/train/weights/best.pt --imgsz 640 --half
```
出力例: `runs/pose/train/weights/best_openvino_model/`
中身: `model.xml`, `model.bin`, `metadata.yaml` など。

### 4.1 INT8 量子化 (任意 / 実験的)
```
python export_openvino.py --weights runs/pose/train/weights/best.pt --int8
```
- 追加で calibration を行うために内部でサンプルを使用
- 精度低下/不安定の可能性 → まず FP16 を検証推奨

---
## 5. 推論 (Inference + Visualization)
### 5.1 画像/ディレクトリ
```
python infer_openvino_pose.py ^
  --model runs/pose/train/weights/best_openvino_model ^
  --source C:\data\pose_test\images\val ^
  --show --conf 0.25
```
結果: `runs/infer_openvino/` に描画済み画像保存。

### 5.2 Webカメラ
```
python infer_openvino_pose.py --model runs/pose/train/weights/best_openvino_model --source 0 --show
```
ESC キーで終了。`result.mp4` 保存。

### 5.3 動画ファイル
```
python infer_openvino_pose.py --model runs/pose/train/weights/best_openvino_model --source input.mp4 --show
```

### 5.4 主なオプション
| オプション | 説明 |
|------------|------|
| `--imgsz` | 推論サイズ (学習と揃えるのが無難) |
| `--conf` | 信頼度閾値 |
| `--kp-radius` | キーポイント描画半径 |
| `--line-thickness` | スケルトン線太さ |
| `--save-dir` | 出力ディレクトリ |

---
## 6. カスタム Keypoints / Skeleton
`train_pose.py` 冒頭の `DEFAULT_KEYPOINTS`, `COCO_SKELETON` を編集。また自動生成 YAML に `keypoint_names` 等を追記可能。
- 変更後はラベル txt の列数と整合することを必ず確認。

---
## 7. オフライン手順 (Windows cmd 例)
### 7.1 事前に `wheels/` に依存を配置後インストール
```
pip install --no-index --find-links=./wheels torch torchvision ultralytics opencv_python
```
(足りない場合はオンライン環境で `pip download <pkg> -d wheels` して再持込)

### 7.2 OpenVINO (未含なら)
```
pip download openvino -d wheels   (オンライン側)
pip install --no-index --find-links=./wheels openvino
```

---
## 8. 精度/速度簡易チェック
1. 学習後: `val/` に対する Ultralytics の `metrics` を標準出力で確認
2. OpenVINO: 推論 FPS (動画/カメラ) のオーバーレイ表示
3. 差異検証: 同一画像セットを `.pt` 版と OpenVINO 版で推論し結果比較 (今後スクリプト追加余地)

---
## 9. よくあるトラブル
| 症状 | 対応 |
|------|------|
| keypoints の数 mismatch | YAML `kpt_shape` とラベル txt 列数を再確認 |
| Export 時にメモリエラー | `--imgsz` を小さく / FP16 (`--half`) を使う |
| 推論で skeleton が描画されない | モデルの `skeleton` 情報が欠落 → YAML / モデル args 確認 |
| FPS 低い | `--imgsz` 縮小, FP16, OpenVINO 最新化 |

---
## 10. 今後の拡張案
- 推論結果 (keypoints 座標) を JSON/CSV へ保存する `--save-json` オプション追加
- バッチ画像推論高速化 (DataLoader 化)
- INT8 量子化後 vs FP16 精度差分レポートスクリプト
- TensorBoard / Weights & Biases ロギング

---
## 11. クイックフローまとめ
```
REM 1) 学習
python train_pose.py --pretrained yolo11n-pose.pt --dataset-root C:\data\pose --epochs 50

REM 2) 変換
python export_openvino.py --weights runs/pose/train/weights/best.pt --imgsz 640 --half

REM 3) 推論
python infer_openvino_pose.py --model runs/pose/train/weights/best_openvino_model --source 0 --show
```

---
何か追加したい項目 (座標保存 / 評価スクリプト 等) があれば指示ください。
