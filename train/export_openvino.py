"""YOLO Pose モデルを OpenVINO IR へエクスポート

Usage:
  python export_openvino.py --weights runs/pose/train/weights/best.pt --imgsz 640 --half
  出力は weights/ ディレクトリ配下に openvino_model/ などとして生成されます。
"""
from __future__ import annotations
import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=str, required=True, help='学習済み .pt パス (best.pt など)')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--device', type=str, default='', help='ONNX 変換段階などで使用するデバイス指定')
    p.add_argument('--half', action='store_true', help='FP16 化 (サポートされる場合)')
    p.add_argument('--int8', action='store_true', help='INT8 量子化 (追加ステップが必要; experimental)')
    p.add_argument('--output-dir', type=str, help='明示的な出力先 (未指定なら weights/ 下)')
    return p.parse_args()


def main():
    args = parse_args()
    w = Path(args.weights)
    if not w.exists():
        raise SystemExit(f'weights が見つかりません: {w}')

    model = YOLO(str(w))

    export_args = dict(format='openvino', imgsz=args.imgsz, device=args.device)
    if args.half:
        export_args['half'] = True
    if args.int8:
        # Ultralytics の INT8 export は additional requirements が必要
        export_args['int8'] = True

    result_path = model.export(**export_args)
    # result_path は出力フォルダへの Path を返す想定 (ultralytics>=8.3.0)

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # 生成物を out_dir へコピー (簡易実装: ユーザーが必要に応じて移動)
        print(f'[INFO] 生成された OpenVINO モデル: {result_path}')
        print('[INFO] 別ディレクトリへの自動コピーが必要なら後で実装してください。')
    else:
        print(f'[INFO] OpenVINO Export 完了: {result_path}')


if __name__ == '__main__':
    main()
