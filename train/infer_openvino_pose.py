"""OpenVINO で YOLO Pose 推論 & 描画

依存:
  pip install openvino==2024.3 opencv-python
  (オフラインの場合は事前に wheel を確保)

Usage:
  python infer_openvino_pose.py --model runs/pose/train/weights/best_openvino_model/ --source sample.jpg
  python infer_openvino_pose.py --model runs/pose/train/weights/best_openvino_model/ --source 0  # WebCam
"""
from __future__ import annotations
import argparse
from pathlib import Path
import time
import cv2
import numpy as np
from ultralytics import YOLO

# NOTE: Ultralytics の OpenVINO export 結果ディレクトリを直接 YOLO(model_path) へ渡すと自動認識します。

PALETTE = (255, 0, 0)

# 正しい COCO 17 keypoints 用スケルトン (1-based index)
COCO_SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7],
    [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5],
    [4, 6], [5, 7]
]

# フォールバック用の簡易 skeleton (直鎖) 17 keypoints 前提
FALLBACK_SKELETON = [[i, i + 1] for i in range(1, 17)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, required=True, help='OpenVINO 変換済みモデルディレクトリ (xml/bins があるフォルダ)')
    p.add_argument('--source', type=str, required=True, help='画像/動画/ディレクトリ あるいは 0 (WebCam)')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--device', type=str, default='')
    p.add_argument('--show', action='store_true')
    p.add_argument('--save-dir', type=str, default='runs/infer_openvino')
    p.add_argument('--line-thickness', type=int, default=2)
    p.add_argument('--kp-radius', type=int, default=3)
    p.add_argument('--force-coco-skeleton', action='store_true', help='常に COCO スケルトンを使用 (17 keypoints 前提)')
    return p.parse_args()


def draw_pose(frame, keypoints, skeleton, color=(0,255,0), radius=3, thickness=2):
    h, w = frame.shape[:2]
    # keypoints: (num_kpts, 3) -> x,y,conf (座標はピクセル)
    for (x, y, c) in keypoints:
        if c > 0.1:
            cv2.circle(frame, (int(x), int(y)), radius, color, -1, lineType=cv2.LINE_AA)
    for a, b in skeleton:
        if a-1 < len(keypoints) and b-1 < len(keypoints):
            xa, ya, ca = keypoints[a-1]
            xb, yb, cb = keypoints[b-1]
            if ca > 0.1 and cb > 0.1:
                cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), color, thickness, lineType=cv2.LINE_AA)
    return frame


def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f'Model path not found: {model_path}')

    model = YOLO(str(model_path))

    def resolve_skeleton(res=None):
        """安全に skeleton を取得し、必要なら COCO に補正。"""
        if args.force_coco_skeleton:
            return COCO_SKELETON
        candidates = []
        if res is not None:
            candidates.append(getattr(res, 'skeleton', None))
        candidates.append(getattr(model, 'skeleton', None))
        inner = getattr(model, 'model', None)
        if inner is not None and not isinstance(inner, str):
            candidates.append(getattr(inner, 'skeleton', None))
            if hasattr(inner, 'args') and isinstance(inner.args, dict):
                candidates.append(inner.args.get('skeleton'))
        if hasattr(model, 'overrides') and isinstance(getattr(model, 'overrides'), dict):
            candidates.append(model.overrides.get('skeleton'))
        # 1つ目の有効候補
        for c in candidates:
            if c:
                sk = c
                break
        else:
            sk = FALLBACK_SKELETON
        # 直鎖のみやエッジ数が 10 未満なら COCO に差し替え (17 keypoints 想定)
        try:
            if isinstance(sk, (list, tuple)):
                edge_cnt = len(sk)
                if edge_cnt < 10 or all((e[1] - e[0] == 1) for e in sk if isinstance(e, (list, tuple)) and len(e) == 2):
                    return COCO_SKELETON
        except Exception:
            pass
        return sk

    # 推論ソース判定
    is_webcam = args.source.isdigit() and len(args.source) <= 2
    source_to_use = 0 if is_webcam else args.source

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if is_webcam or Path(source_to_use).is_file() and source_to_use.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        cap = cv2.VideoCapture(source_to_use)
        if not cap.isOpened():
            raise SystemExit(f'Failed to open source: {source_to_use}')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = save_dir / 'result.mp4'
        writer = None
        prev = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            res = model.predict(source=frame, imgsz=args.imgsz, conf=args.conf, verbose=False, device=args.device)[0]
            if res.keypoints is not None:
                kpts = res.keypoints.xy.cpu().numpy()  # (num_instances, num_kpts, 2)
                kconfs = res.keypoints.conf.cpu().numpy()  # (num_instances, num_kpts)
                skel = resolve_skeleton(res)
                for i in range(kpts.shape[0]):
                    merged = np.concatenate([kpts[i], kconfs[i][..., None]], axis=-1)
                    draw_pose(frame, merged, skel, color=PALETTE, radius=args.kp_radius, thickness=args.line_thickness)
            fps = 1.0 / (time.time() - prev)
            prev = time.time()
            cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2, cv2.LINE_AA)
            if args.show:
                cv2.imshow('pose', frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            if writer is None:
                writer = cv2.VideoWriter(str(out_path), fourcc, 30, (frame.shape[1], frame.shape[0]))
            writer.write(frame)
        cap.release()
        if writer:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
        print(f'[INFO] Saved video to {out_path}')
    else:
        # 画像 or ディレクトリ
        sources = []
        p = Path(source_to_use)
        if p.is_dir():
            for ext in ('*.jpg','*.jpeg','*.png','*.bmp'):
                sources.extend(p.glob(ext))
        else:
            sources = [p]
        for img_path in sources:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f'[WARN] Failed to read {img_path}')
                continue
            res = model.predict(source=img, imgsz=args.imgsz, conf=args.conf, verbose=False, device=args.device)[0]
            if res.keypoints is not None:
                kpts = res.keypoints.xy.cpu().numpy()
                kconfs = res.keypoints.conf.cpu().numpy()
                skel = resolve_skeleton(res)
                for i in range(kpts.shape[0]):
                    merged = np.concatenate([kpts[i], kconfs[i][..., None]], axis=-1)
                    draw_pose(img, merged, skel, color=PALETTE, radius=args.kp_radius, thickness=args.line_thickness)
            out_file = save_dir / img_path.name
            cv2.imwrite(str(out_file), img)
            print(f'[INFO] saved {out_file}')


if __name__ == '__main__':
    main()
