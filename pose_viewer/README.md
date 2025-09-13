# pose_viewer (Minimal OpenVINO + tkinter)

## 概要
OpenVINO 変換済みの YOLO Pose (IR: XML/BIN) を使い、Webカメラ映像にキーポイントとスケルトンを描画する最小GUIアプリです。

## 構成
- `main.py` : エントリポイント（設定読み込み、GUI起動、キャプチャ制御）
- `backend_openvino.py` : OpenVINO 推論バックエンド（前処理・推論・後処理）
- `gui.py` : tkinter GUI
- `pose_draw.py` : 骨格描画ユーティリティ
- `settings.py` : 設定の型・検証・デフォルト適用
- `config.yaml` : 設定ファイル
- `models/` : OpenVINO IR配置（例: `test.xml` と `test.bin`）

## 依存
`requirements.txt` を参照してください。

## 設定（抜粋）
`pose_viewer/config.yaml`
- `model.path` : IR の XML パス
- `model.device` : `CPU` など
- `model.img_size` : 推論の入力解像度（正方）
- `input.source` : `0`（Webカメラ）/ 画像・動画パス / URL
- `input.capture.width|height|fps` : 任意でキャプチャ設定
- `performance.skip` : Nフレームごとに1回処理
- `postprocess.det_confidence` : バウンディングボックスのしきい値
- `postprocess.kpt_confidence` : キーポイントの描画しきい値
- `postprocess.max_detections` : 1フレームの最大人数
- `postprocess.nms_iou` : NMS の IoU しきい値（重複抑制）
- `draw.skeleton` : スケルトン描画の有無
- `draw.keypoint_radius` / `draw.thickness` : 描画サイズ
- `display.width|height` : GUI 表示領域サイズ

## 実行
`pose_viewer` ディレクトリをカレントにして実行:
```
python main.py
```
