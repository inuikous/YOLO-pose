"""Selective small COCO person keypoints downloader + YOLO Pose format converter.

Purpose:
  Download only a small number of COCO 2017 images containing person keypoints
  (without fetching the entire ~19GB train set) and convert to YOLO pose format.

Features:
  - Downloads annotations (if missing) only once (annotations_trainval2017.zip)
  - Randomly selects N train images and M val images that contain at least 1 person with keypoints
  - Downloads only those JPEGs via individual HTTP GET requests
  - Converts to YOLO Pose labels (17 keypoints, class=person=0)
  - Generates minimal pose.yaml

Usage example:
  python download_coco_small_pose.py \
    --root C:/datasets/coco_mini_pose \
    --train-count 300 --val-count 60 --seed 42

Then train:
  python train_pose.py --pretrained yolo11n-pose.pt --data C:/datasets/coco_mini_pose/pose.yaml --epochs 10 --batch 16 --imgsz 640

NOTE:
  Requires internet access. If a request fails, script retries a few times then skips image.
  You must respect COCO dataset terms of use.
"""
from __future__ import annotations
import argparse
import json
import random
import time
from pathlib import Path
import zipfile
import io
import sys

try:
    import requests  # Provided in wheels/requests
except ImportError:  # fallback
    requests = None  # type: ignore

ANN_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
TRAIN_BASE_URL = "http://images.cocodataset.org/train2017"
VAL_BASE_URL = "http://images.cocodataset.org/val2017"
ANN_TRAIN_FILE = "annotations/person_keypoints_train2017.json"
ANN_VAL_FILE = "annotations/person_keypoints_val2017.json"

SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7],
    [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5],
    [4, 6], [5, 7]
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='Output root directory')
    ap.add_argument('--train-count', type=int, default=300)
    ap.add_argument('--val-count', type=int, default=60)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-retries', type=int, default=3)
    ap.add_argument('--timeout', type=float, default=15.0)
    ap.add_argument('--no-val', action='store_true', help='Do not download val; duplicate train for val')
    return ap.parse_args()


def ensure_requests():
    if requests is None:
        print('[ERROR] requests not available. Install requests first.')
        sys.exit(1)


def download_annotations(root: Path) -> tuple[Path, Path]:
    ann_dir = root / 'annotations'
    train_json = ann_dir / 'person_keypoints_train2017.json'
    val_json = ann_dir / 'person_keypoints_val2017.json'
    if train_json.exists() and val_json.exists():
        return train_json, val_json
    ann_dir.mkdir(parents=True, exist_ok=True)
    print('[INFO] Downloading annotations zip ...')
    r = requests.get(ANN_ZIP_URL, timeout=60)
    r.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    for name in zf.namelist():
        if name.endswith('person_keypoints_train2017.json') or name.endswith('person_keypoints_val2017.json'):
            target = ann_dir / Path(name).name
            with zf.open(name) as src, open(target, 'wb') as dst:
                dst.write(src.read())
    if not train_json.exists():
        raise FileNotFoundError('Train keypoints JSON missing after extraction.')
    if not val_json.exists():
        raise FileNotFoundError('Val keypoints JSON missing after extraction.')
    print('[INFO] Annotations downloaded.')
    return train_json, val_json


def load_coco(json_path: Path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    images = {img['id']: img for img in data['images']}
    anns_by_image = {}
    for ann in data['annotations']:
        if ann.get('iscrowd', 0) == 1:
            continue
        if ann.get('num_keypoints', 0) == 0:
            continue
        anns_by_image.setdefault(ann['image_id'], []).append(ann)
    return images, anns_by_image


def pick_images(anns_by_image, count: int) -> list[int]:
    ids = list(anns_by_image.keys())
    random.shuffle(ids)
    return ids[:count]


def http_get(url: str, dst: Path, retries: int, timeout: float):
    if dst.exists():
        return True
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200 and resp.content:
                with open(dst, 'wb') as f:
                    f.write(resp.content)
                return True
            else:
                print(f'[WARN] status {resp.status_code} for {url}')
        except Exception as e:
            print(f'[WARN] attempt {attempt} failed: {e}')
        time.sleep(1.0 * attempt)
    print(f'[ERROR] Failed to download {url}')
    return False


def write_label(label_path: Path, anns, img_w: int, img_h: int):
    lines = []
    for ann in anns:
        x, y, w, h = ann['bbox']
        if w <= 0 or h <= 0:
            continue
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        nw = w / img_w
        nh = h / img_h
        kpts = ann.get('keypoints', [])
        if len(kpts) != 17 * 3:
            continue
        parts = []
        for i in range(17):
            kx = kpts[3 * i]
            ky = kpts[3 * i + 1]
            kv = kpts[3 * i + 2]
            if kx == 0 and ky == 0:
                parts.extend([0.0, 0.0, 0])
            else:
                parts.extend([kx / img_w, ky / img_h, kv])
        line = '0 ' + ' '.join([
            f'{cx:.6f}', f'{cy:.6f}', f'{nw:.6f}', f'{nh:.6f}'
        ] + [f'{v:.6f}' if (i % 3) != 2 else str(int(v)) for i, v in enumerate(parts)])
        lines.append(line)
    if lines:
        label_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def ensure_dirs(root: Path):
    for sub in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        (root / sub).mkdir(parents=True, exist_ok=True)


def write_yaml(root: Path):
    yaml_path = root / 'pose.yaml'
    yaml_path.write_text(
        f"""# Small COCO subset pose dataset\npath: {root.as_posix()}\ntrain: images/train\nval: images/val\nnc: 1\nnames: [person]\nkpt_shape: [17, 3]\nskeleton: {SKELETON}\n""",
        encoding='utf-8'
    )
    return yaml_path


def process_split(root: Path, split_name: str, image_ids: list[int], images, anns_by_image, base_url: str, retries: int, timeout: float):
    for img_id in image_ids:
        info = images[img_id]
        file_name = info['file_name']  # 000000123456.jpg
        img_w = info['width']; img_h = info['height']
        url = f"{base_url}/{file_name}"
        out_img = root / 'images' / split_name / file_name
        out_lbl = root / 'labels' / split_name / (file_name.replace('.jpg', '.txt'))
        ok = http_get(url, out_img, retries, timeout)
        if not ok:
            continue
        anns = anns_by_image.get(img_id, [])
        write_label(out_lbl, anns, img_w, img_h)


def duplicate_train_to_val(root: Path):
    import shutil
    train_imgs = list((root / 'images' / 'train').glob('*.jpg'))
    for p in train_imgs:
        dst = root / 'images' / 'val' / p.name
        if not dst.exists():
            shutil.copy2(p, dst)
    train_lbls = list((root / 'labels' / 'train').glob('*.txt'))
    for p in train_lbls:
        dst = root / 'labels' / 'val' / p.name
        if not dst.exists():
            shutil.copy2(p, dst)


def main():
    args = parse_args()
    ensure_requests()
    random.seed(args.seed)
    root = Path(args.root)
    ensure_dirs(root)

    train_json, val_json = download_annotations(root)

    train_images, train_anns_by_image = load_coco(train_json)
    train_ids_all = list(train_anns_by_image.keys())
    random.shuffle(train_ids_all)
    sel_train = train_ids_all[: args.train_count]
    print(f'[INFO] Selected train images: {len(sel_train)}')

    process_split(root, 'train', sel_train, train_images, train_anns_by_image, TRAIN_BASE_URL, args.max_retries, args.timeout)

    if not args.no_val:
        val_images, val_anns_by_image = load_coco(val_json)
        val_ids_all = list(val_anns_by_image.keys())
        random.shuffle(val_ids_all)
        sel_val = val_ids_all[: args.val_count]
        print(f'[INFO] Selected val images: {len(sel_val)}')
        process_split(root, 'val', sel_val, val_images, val_anns_by_image, VAL_BASE_URL, args.max_retries, args.timeout)
    else:
        duplicate_train_to_val(root)

    yaml_path = write_yaml(root)
    print('[INFO] Done.')
    print(f'[INFO] YAML: {yaml_path}')
    print('Train example:')
    print(f'  python train_pose.py --pretrained yolo11n-pose.pt --data {yaml_path} --epochs 10 --batch 16 --imgsz 640')


if __name__ == '__main__':
    main()
