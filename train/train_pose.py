"""YOLO Pose 再学習スクリプト

Usage (例):
  python train_pose.py --pretrained yolo11n-pose.pt --dataset-root C:/data/pose --epochs 50 --imgsz 640
  既に data YAML を持っている場合:
  python train_pose.py --pretrained yolo11n-pose.pt --data C:/data/pose/pose.yaml

オフライン環境: 事前に wheels をインストール済みで ultralytics が使えること。
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from ultralytics import YOLO

DEFAULT_KEYPOINTS = 17  # COCO
DEFAULT_NAMES = [
    'person'
]
# COCO skeleton (pairs of keypoint indices (1-based) ); ultralytics YAML expects 1-based?
COCO_SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7],
    [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5],
    [4, 6], [5, 7]
]

def generate_pose_yaml(dataset_root: Path, keypoints: int, overwrite: bool = False) -> Path:
    """dataset_root 下に train/val フォルダ構成がある前提で pose.yaml を生成。
    期待構成:
      dataset_root/
        images/train/*.jpg
        images/val/*.jpg
        labels/train/*.txt
        labels/val/*.txt
    ラベルフォーマット: class x y w h x1 y1 v1 x2 y2 v2 ... (正規化)  v=0/1/2
    """
    yaml_path = dataset_root / 'pose.yaml'
    if yaml_path.exists() and not overwrite:
        return yaml_path

    names_list = DEFAULT_NAMES  # 単一クラス想定。複数ある場合は編集してください。
    # keypoints 配列: ここでは COCO 名称を省略し index 文字列にする
    kpt_shape = [keypoints, 3]
    yaml_text = f"""# Auto-generated pose dataset YAML
path: {dataset_root.as_posix()}
train: images/train
val: images/val
# test: images/test
nc: {len(names_list)}
names: {names_list}
# keypoints 設定
kpt_shape: {kpt_shape}  # [num_keypoints, dim]
skeleton: {COCO_SKELETON}
# 追加で keypoints 名称を付けたい場合は下記を編集
# keypoint_names: ['nose','left_eye',...]
"""
    yaml_path.write_text(yaml_text, encoding='utf-8')
    return yaml_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pretrained', type=str, default='yolo11n-pose.pt', help='事前学習モデル(.pt)')
    p.add_argument('--data', type=str, help='既存 pose.yaml へのパス')
    p.add_argument('--dataset-root', type=str, help='pose.yaml を自動生成したいルート (images/, labels/ が存在)')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--device', type=str, default='', help='例: 0 or 0,1 or cpu (空なら自動)')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--project', type=str, default='runs/pose')
    p.add_argument('--name', type=str, default='train')
    p.add_argument('--exist-ok', action='store_true')
    p.add_argument('--keypoints', type=int, default=17, help='自動生成時の keypoints 数')
    p.add_argument('--overwrite-yaml', action='store_true', help='既存 pose.yaml を上書き')
    p.add_argument('--lr0', type=float, help='初期学習率 (未指定ならデフォルト)')
    return p.parse_args()


def main():
    args = parse_args()

    if not args.data:
        if not args.dataset_root:
            raise SystemExit('--data か --dataset-root のどちらかを指定してください')
        root = Path(args.dataset_root)
        if not (root / 'images' / 'train').exists():
            raise SystemExit(f'images/train が見つかりません: {root}')
        args.data = str(generate_pose_yaml(root, args.keypoints, overwrite=args.overwrite_yaml))
        print(f'[INFO] pose.yaml 生成: {args.data}')
    else:
        if not Path(args.data).exists():
            raise SystemExit(f'data YAML が存在しません: {args.data}')

    if not Path(args.pretrained).exists():
        print(f'[WARN] Pretrained {args.pretrained} が見つかりません。Ultralytics が自動取得を試みます。')

    model = YOLO(args.pretrained)

    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        task='pose'
    )
    if args.lr0 is not None:
        train_kwargs['lr0'] = args.lr0

    print('[INFO] Training start with args:')
    for k, v in train_kwargs.items():
        print(f'  {k}: {v}')

    results = model.train(**train_kwargs)
    print('[INFO] Training finished.')
    print(results)
    print('[INFO] 最良モデル: runs/pose/.../weights/best.pt を確認してください')


if __name__ == '__main__':
    main()
