"""Drawing utilities placeholder."""

import cv2

def draw_pose(image, keypoints, skeleton=None, radius=3, thickness=2):
    # keypoints: list of (x,y,score)
    for (x,y,score) in keypoints:
        cv2.circle(image, (int(x), int(y)), radius, (0,255,0), -1)
    if skeleton:
        for a,b in skeleton:
            if a < len(keypoints) and b < len(keypoints):
                xa,ya,_ = keypoints[a]
                xb,yb,_ = keypoints[b]
                cv2.line(image, (int(xa), int(ya)), (int(xb), int(yb)), (255,0,0), thickness)
    return image
