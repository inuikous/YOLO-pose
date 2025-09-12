"""小規模ダミー Pose データセット生成スクリプト

目的:
  COCO を用意できない環境でも学習パイプライン確認用に最小データを作る。

生成物構成 (例: --out dummy_pose_dataset):
  dummy_pose_dataset/
    images/train/*.jpg
    images/val/*.jpg
    labels/train/*.txt
    labels/val/*.txt
    pose.yaml

各画像: 640x640 の単色背景に 1 人分の単純なスティックフィギュア。
Keypoints: COCO 17 点 (順序: nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle) を簡易配置。

Usage:
  python generate_dummy_pose_dataset.py --out dummy_pose_dataset --num-images 40 --val-ratio 0.2
  その後:
  python train_pose.py --pretrained yolo11n-pose.pt --data dummy_pose_dataset/pose.yaml --epochs 1 --imgsz 640
"""
from __future__ import annotations
import argparse
from pathlib import Path
import random
import cv2
import numpy as np

SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7],
    [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5],
    [4, 6], [5, 7]
]

KPT_COUNT = 17
IMG_SIZE = 640


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=str, default='dummy_pose_dataset')
    ap.add_argument('--num-images', type=int, default=40)
    ap.add_argument('--val-ratio', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    return ap.parse_args()


def ensure_dirs(root: Path):
    for p in [root / 'images' / 'train', root / 'images' / 'val', root / 'labels' / 'train', root / 'labels' / 'val']:
        p.mkdir(parents=True, exist_ok=True)


def gen_keypoints_and_bbox():
    # ランダムに人物中心とスケールを決め、そこから相対配置
    cx = random.randint(200, 440)
    cy = random.randint(220, 460)
    scale = random.randint(120, 180)
    # 簡易骨格相対座標 (ベース) [x,y]
    rel = {
        'nose': (0, -0.45),
        'left_eye': (-0.05, -0.47), 'right_eye': (0.05, -0.47),
        'left_ear': (-0.12, -0.45), 'right_ear': (0.12, -0.45),
        'left_shoulder': (-0.18, -0.30), 'right_shoulder': (0.18, -0.30),
        'left_elbow': (-0.28, -0.05), 'right_elbow': (0.28, -0.05),
        'left_wrist': (-0.30, 0.20), 'right_wrist': (0.30, 0.20),
        'left_hip': (-0.12, 0.05), 'right_hip': (0.12, 0.05),
        'left_knee': (-0.10, 0.35), 'right_knee': (0.10, 0.35),
        'left_ankle': (-0.10, 0.60), 'right_ankle': (0.10, 0.60)
    }
    order = ['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle']
    kpts = []
    xs, ys = [], []
    for name in order:
        rx, ry = rel[name]
        x = int(cx + rx * scale)
        y = int(cy + ry * scale)
        x = max(0, min(IMG_SIZE - 1, x))
        y = max(0, min(IMG_SIZE - 1, y))
        xs.append(x); ys.append(y)
        kpts.append((x, y, 2))  # visibility=2
    # bbox 計算
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = x_max - x_min + 1
    h = y_max - y_min + 1
    cx_box = x_min + w / 2
    cy_box = y_min + h / 2
    return kpts, (cx_box, cy_box, w, h)


def save_label(path: Path, kpts, bbox):
    cx_box, cy_box, w, h = bbox
    cx_n = cx_box / IMG_SIZE
    cy_n = cy_box / IMG_SIZE
    w_n = w / IMG_SIZE
    h_n = h / IMG_SIZE
    flat = []
    for x, y, v in kpts:
        flat.extend([x / IMG_SIZE, y / IMG_SIZE, v])
    parts = ["0", f"{cx_n:.6f}", f"{cy_n:.6f}", f"{w_n:.6f}", f"{h_n:.6f}"]
    for i, val in enumerate(flat):
        if i % 3 == 2:  # visibility integer
            parts.append(str(int(val)))
        else:
            parts.append(f"{val:.6f}")
    path.write_text(" ".join(parts) + "\n", encoding='utf-8')


def draw_figure(img, kpts):
    # スケルトン描画
    for a, b in SKELETON:
        a -= 1; b -= 1
        if 0 <= a < len(kpts) and 0 <= b < len(kpts):
            xa, ya, _ = kpts[a]
            xb, yb, _ = kpts[b]
            cv2.line(img, (xa, ya), (xb, yb), (0, 255, 0), 2, cv2.LINE_AA)
    for x, y, v in kpts:
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1, cv2.LINE_AA)


def write_yaml(root: Path):
    yaml_path = root / 'pose.yaml'
    yaml_path.write_text(
        f"""# Dummy pose dataset YAML\npath: {root.as_posix()}\ntrain: images/train\nval: images/val\nnc: 1\nnames: [person]\nkpt_shape: [17, 3]\nskeleton: {SKELETON}\n""",
        encoding='utf-8'
    )
    return yaml_path


def main():
    args = parse_args()
    random.seed(args.seed)
    root = Path(args.out)
    ensure_dirs(root)

    n_val = max(1, int(args.num_images * args.val_ratio))
    n_train = args.num_images - n_val
    indices = list(range(args.num_images))
    random.shuffle(indices)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    for split, id_list in [('train', train_idx), ('val', val_idx)]:
        for i in id_list:
            kpts, bbox = gen_keypoints_and_bbox()
            img = np.full((IMG_SIZE, IMG_SIZE, 3), 255, dtype=np.uint8)
            draw_figure(img, kpts)
            img_path = root / 'images' / split / f'{i:05d}.jpg'
            lbl_path = root / 'labels' / split / f'{i:05d}.txt'
            cv2.imwrite(str(img_path), img)
            save_label(lbl_path, kpts, bbox)
    yaml_path = write_yaml(root)
    print(f'[INFO] Generated dummy dataset at: {root}')
    print(f'[INFO] YAML: {yaml_path}')
    print('Train command example:')
    print(f'  python train_pose.py --pretrained yolo11n-pose.pt --data {yaml_path} --epochs 1 --imgsz 640')


if __name__ == '__main__':
    main()
