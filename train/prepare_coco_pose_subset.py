"""COCO person keypoints から小規模サブセットを YOLO Pose 形式へ変換

Usage (例):
  python prepare_coco_pose_subset.py \
    --train-ann D:/coco/annotations/person_keypoints_train2017.json \
    --val-ann D:/coco/annotations/person_keypoints_val2017.json \
    --train-images D:/coco/train2017 \
    --val-images D:/coco/val2017 \
    --out D:/datasets/coco_person_subset \
    --train-count 200 --val-count 50 --copy-images

結果構成:
  <out>/
    images/train/*.jpg
    images/val/*.jpg
    labels/train/*.txt
    labels/val/*.txt
    pose.yaml  (自動生成)

制限:
- 単一クラス (person) のみ。
- Keypoints は COCO 17 個。

COCO → YOLO Pose ラベル行:
  class cx cy w h x1 y1 v1 x2 y2 v2 ... (正規化)

"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
import shutil
from typing import Dict, List, Any

# COCO skeleton (1-based index)
SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7],
    [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5],
    [4, 6], [5, 7]
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train-ann', required=True, help='person_keypoints_train2017.json')
    p.add_argument('--val-ann', help='person_keypoints_val2017.json (任意)')
    p.add_argument('--train-images', required=True, help='train2017 画像ディレクトリ')
    p.add_argument('--val-images', help='val2017 画像ディレクトリ (val 使用時)')
    p.add_argument('--out', required=True, help='出力ルート')
    p.add_argument('--train-count', type=int, default=200, help='抽出する train 画像数')
    p.add_argument('--val-count', type=int, default=50, help='抽出する val 画像数')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--copy-images', action='store_true', help='画像をコピー (未指定なら元パス参照のまま symlink 試行 -> 失敗ならコピー)')
    p.add_argument('--no-val', action='store_true', help='val を作らず train のみ')
    return p.parse_args()


def load_coco(ann_path: Path) -> Dict[str, Any]:
    with ann_path.open('r', encoding='utf-8') as f:
        return json.load(f)


def index_coco(data: Dict[str, Any]):
    images = {img['id']: img for img in data['images']}
    anns_by_image = {}
    for ann in data['annotations']:
        if ann.get('iscrowd', 0) == 1:
            continue
        if ann.get('num_keypoints', 0) == 0:
            continue
        anns_by_image.setdefault(ann['image_id'], []).append(ann)
    return images, anns_by_image


def select_images(images: Dict[int, Dict[str, Any]], anns_by_image, count: int) -> List[int]:
    valid_ids = [imid for imid, anns in anns_by_image.items() if len(anns) > 0]
    if count > len(valid_ids):
        count = len(valid_ids)
    return valid_ids[:count]


def write_label_file(out_label_path: Path, anns: List[Dict[str, Any]], img_w: int, img_h: int):
    lines = []
    for ann in anns:
        # bbox: [x,y,w,h] (左上原点)
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
        kp_parts = []
        for i in range(17):
            kx = kpts[3 * i]
            ky = kpts[3 * i + 1]
            kv = kpts[3 * i + 2]  # 0/1/2
            if kx == 0 and ky == 0:
                # 欠損扱い
                kp_parts.extend([0.0, 0.0, 0])
            else:
                kp_parts.extend([kx / img_w, ky / img_h, kv])
        line = '0 ' + ' '.join(
            [f'{cx:.6f}', f'{cy:.6f}', f'{nw:.6f}', f'{nh:.6f}'] + [f'{v:.6f}' if (i % 3) != 2 else f'{int(v)}' for i, v in enumerate(kp_parts)]
        )
        lines.append(line)
    if lines:
        out_label_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def copy_or_link(src: Path, dst: Path, force_copy: bool):
    if dst.exists():
        return
    if not force_copy:
        try:
            # Windows では symlink には管理者権限/開発者モードが必要な場合あり
            dst.symlink_to(src)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)


def build_yaml(root: Path):
    yaml_path = root / 'pose.yaml'
    text = f"""# Generated minimal pose dataset YAML
path: {root.as_posix()}
train: images/train
val: images/val
nc: 1
names: [person]
kpt_shape: [17, 3]
skeleton: {SKELETON}
"""
    yaml_path.write_text(text, encoding='utf-8')
    return yaml_path


def process_split(name: str, image_ids: List[int], images: Dict[int, Dict[str, Any]], anns_by_image, images_dir: Path, out_root: Path, copy_images: bool):
    out_img_dir = out_root / 'images' / name
    out_lbl_dir = out_root / 'labels' / name
    ensure_dir(out_img_dir)
    ensure_dir(out_lbl_dir)
    for imid in image_ids:
        img_info = images[imid]
        file_name = img_info['file_name']
        width, height = img_info['width'], img_info['height']
        src_img = images_dir / file_name
        if not src_img.exists():
            # 一部 images のみダウンロードしているケースで欠損をスキップ
            continue
        dst_img = out_img_dir / file_name
        copy_or_link(src_img, dst_img, copy_images)
        label_file = out_lbl_dir / (Path(file_name).stem + '.txt')
        anns = anns_by_image.get(imid, [])
        write_label_file(label_file, anns, width, height)


def main():
    args = parse_args()
    random.seed(args.seed)

    out_root = Path(args.out)
    ensure_dir(out_root)

    train_data = load_coco(Path(args.train_ann))
    train_images, train_anns_by_image = index_coco(train_data)

    # 画像 ID をシャッフルしてから選択
    train_valid_ids = [imid for imid in train_anns_by_image.keys()]
    random.shuffle(train_valid_ids)
    train_sel = select_images(train_images, train_anns_by_image, args.train_count)

    process_split('train', train_sel, train_images, train_anns_by_image, Path(args.train_images), out_root, args.copy_images)

    if not args.no_val and args.val_ann and args.val_images:
        val_data = load_coco(Path(args.val_ann))
        val_images, val_anns_by_image = index_coco(val_data)
        val_valid_ids = [imid for imid in val_anns_by_image.keys()]
        random.shuffle(val_valid_ids)
        val_sel = select_images(val_images, val_anns_by_image, args.val_count)
        process_split('val', val_sel, val_images, val_anns_by_image, Path(args.val_images), out_root, args.copy_images)
    else:
        # val 無しの場合は train を val に再利用 (簡易)
        (out_root / 'images' / 'val').mkdir(parents=True, exist_ok=True)
        (out_root / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
        # シンボリックリンクまたはコピー
        for f in (out_root / 'images' / 'train').glob('*'):
            copy_or_link(f, (out_root / 'images' / 'val' / f.name), True)
        for f in (out_root / 'labels' / 'train').glob('*'):
            copy_or_link(f, (out_root / 'labels' / 'val' / f.name), True)

    yaml_path = build_yaml(out_root)
    print(f'[INFO] Subset 完了: {out_root}')
    print(f'[INFO] YAML: {yaml_path}')
    print('学習例:')
    print(f'  python train_pose.py --pretrained yolo11n-pose.pt --data {yaml_path} --epochs 10 --imgsz 640')


if __name__ == '__main__':
    main()
