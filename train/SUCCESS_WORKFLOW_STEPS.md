# YOLO Pose (小規模COCOサブセット) 成功手順まとめ

本ファイルは今回実際に成功した一連の最小～基本フローを整理したものです。

## 0. 前提環境
- Python venv 有効化済み: `C:/exe/YOLO-pose/venv/`
- インストール済: `ultralytics 8.3.196`, `torch 2.8.0+cpu` (CPU 実行)
- 事前に `yolo11n-pose.pt` を配置済

## 1. 少量 COCO キーポイントサブセット取得
スクリプト: `download_coco_small_pose.py`
```
python download_coco_small_pose.py --root C:/datasets/coco_mini_pose --train-count 300 --val-count 60 --seed 42
```
出力:
- 画像: `C:/datasets/coco_mini_pose/images/train|val/*.jpg`
- ラベル: `C:/datasets/coco_mini_pose/labels/train|val/*.txt`
- YAML: `C:/datasets/coco_mini_pose/pose.yaml`

## 2. 学習 (10 epoch, CPU)
スクリプト: `train_pose.py`
```
python train_pose.py --pretrained yolo11n-pose.pt --data C:/datasets/coco_mini_pose/pose.yaml --epochs 10 --batch 8 --imgsz 640 --exist-ok
```
主な出力:
- 重み: `runs/pose/train/weights/best.pt`
- メトリクス例 (今回):
  - Pose Precision ≈ 0.82
  - Pose Recall ≈ 0.43
  - Pose mAP50 ≈ 0.47
  - Pose mAP50-95 ≈ 0.19

改善余地: 画像枚数増 / epoch 増 / GPU 利用 / LR 調整 / データ拡張。

## 3. OpenVINO 変換
スクリプト: `export_openvino.py`
```
python export_openvino.py --weights runs/pose/train/weights/best.pt --imgsz 640 --half
```
出力ディレクトリ例:
```
runs/pose/train/weights/best_openvino_model/
  model.xml
  model.bin
  metadata.yaml
```

## 4. OpenVINO 推論 (画像ディレクトリ)
スクリプト: `infer_openvino_pose.py`
```
python infer_openvino_pose.py --model runs/pose/train/weights/best_openvino_model --source C:/datasets/coco_mini_pose/images/val --show
```
保存先 (指定しない場合既定): `runs/infer_openvino` (修正テスト時は `runs/infer_openvino_fix_test` を使用)

### Skeleton 取得エラー対策
`infer_openvino_pose.py` にフォールバック skeleton (`FALLBACK_SKELETON`) を実装済み。

## 5. CPU 実行について
- `torch 2.8.0+cpu` のため GPU 利用不可 → ログの `GPU_mem 0G` は正常。
- GPU を使いたい場合は CUDA 対応 torch を再インストールし、`--device 0` 指定。

## 6. 追加スクリプト概要
| ファイル | 用途 |
|----------|------|
| `download_coco_small_pose.py` | 必要枚数のみ COCO からオンライン取得 & YOLO 変換 |
| `train_pose.py` | Pose 再学習 (YAML 自動生成機能付き) |
| `export_openvino.py` | 学習済み `.pt` → OpenVINO IR 変換 |
| `infer_openvino_pose.py` | OpenVINO 推論 + 骨格描画 (fallback skeleton) |
| `prepare_coco_pose_subset.py` | オフラインでフル COCO から部分抽出 (今回は未使用) |
| `generate_dummy_pose_dataset.py` | ダミー生成 (初期検証用) |

## 7. 典型トラブル / 対処
| 症状 | 対処 |
|------|------|
| `AttributeError: 'str' object has no attribute 'args'` | 推論スクリプト更新版使用 (skeleton 取得部) |
| GPU が 0G | CPU ビルド / CUDA 版再インストールで解決 |
| 推論結果で線が出ない | 閾値 `--conf` を下げる / skeleton fallback 確認 |
| 学習が遅い | 画像数削減 / `--batch` 調整 / GPU 導入 |

## 8. 次の拡張候補
- 推論結果の JSON / CSV 出力 (`--save-json`) 追加
- INT8 量子化 export の検証
- mAP 比較スクリプト (PyTorch vs OpenVINO)
- 追加データの段階的学習 (fine-tune)

## 9. 最短クイックまとめ
```
# 1) 小規模COCO取得
python download_coco_small_pose.py --root C:/datasets/coco_mini_pose --train-count 300 --val-count 60
# 2) 学習
python train_pose.py --pretrained yolo11n-pose.pt --data C:/datasets/coco_mini_pose/pose.yaml --epochs 10 --batch 8 --imgsz 640
# 3) 変換
python export_openvino.py --weights runs/pose/train/weights/best.pt --imgsz 640 --half
# 4) 推論
python infer_openvino_pose.py --model runs/pose/train/weights/best_openvino_model --source C:/datasets/coco_mini_pose/images/val --show
```

---
これで現状の成功手順は網羅しています。追加したい観点があれば指示ください。
