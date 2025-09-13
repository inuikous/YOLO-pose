"""ポーズ描画ユーティリティ（最小）。"""

import cv2

def draw_pose(image, keypoints, skeleton=None, radius=3, thickness=2):
    """キーポイントとスケルトンを画像に描画します。

    Args:
        image: BGR 画像（``numpy.ndarray``）。
        keypoints: ``[(x, y, score), ...]`` の配列。
        skeleton: スケルトンの接続を示すタプルのリスト。例: ``[(16,14), (14,12), ...]``。
        radius: キーポイント円の半径。
        thickness: 線の太さ。

    Returns:
        numpy.ndarray: 描画後の BGR 画像。
    """
    # キーポイント: list of (x, y, score)
    for (x, y, score) in keypoints:
        cv2.circle(image, (int(x), int(y)), radius, (0, 255, 0), -1)
    if skeleton:
        for a, b in skeleton:
            if a < len(keypoints) and b < len(keypoints):
                xa, ya, _ = keypoints[a]
                xb, yb, _ = keypoints[b]
                cv2.line(image, (int(xa), int(ya)), (int(xb), int(yb)), (255, 0, 0), thickness)
    return image
