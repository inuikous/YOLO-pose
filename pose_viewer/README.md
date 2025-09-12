# pose_viewer (Minimal OpenVINO + tkinter)

## 概要
OpenVINO 変換済み YOLO Pose モデル (XML+BIN) を用いて、Webカメラ映像に骨格描画する簡易GUIアプリ。

## 構成
- `main.py` : エントリポイント (後で実装)
- `backend_openvino.py` : OpenVINO推論バックエンド
- `gui.py` : tkinter GUI
- `pose_draw.py` : 描画ユーティリティ
- `config.yaml` : 設定
- `models/` : OpenVINO IRファイル配置 (例: yolov8n-pose.xml / .bin)

## 依存
`requirements.txt` 参照

## 次の実装予定
1. OpenVINO モデルロード
2. 推論ループ + スレッド
3. GUI へ画像転送
