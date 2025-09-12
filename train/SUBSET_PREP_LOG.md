# COCO サブセット準備 作業ログ

## 追加ファイル
- `prepare_coco_pose_subset.py` : COCO person keypoints JSON から小規模サブセットを YOLO Pose 形式へ変換するスクリプト

## 主な機能
- train/val 画像枚数を指定 (`--train-count`, `--val-count`)
- person keypoints のみ抽出 (num_keypoints>0, iscrowd=0)
- YOLO Pose ラベル (bbox + 17*3 keypoints) を生成
- 画像をコピー or symlink (Windows で symlink 失敗時はコピー)
- val 未指定時は train を複製して val を構成
- 最小 `pose.yaml` を自動生成

## 使用例
```
python prepare_coco_pose_subset.py ^
  --train-ann D:/coco/annotations/person_keypoints_train2017.json ^
  --val-ann D:/coco/annotations/person_keypoints_val2017.json ^
  --train-images D:/coco/train2017 ^
  --val-images D:/coco/val2017 ^
  --out D:/datasets/coco_person_subset ^
  --train-count 200 --val-count 50 --copy-images
```

その後学習:
```
python train_pose.py --pretrained yolo11n-pose.pt --data D:/datasets/coco_person_subset/pose.yaml --epochs 10 --imgsz 640
```

## 次に可能な拡張 (未実装)
- 画像フィルタ (解像度閾値など)
- ランダム seed 変更による複数サンプル生成
- クラス拡張 (他クラス追加) / multi-class 対応
- stats 出力 (平均 bbox サイズ, keypoint 有効率)
