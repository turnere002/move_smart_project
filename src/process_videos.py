# src/process_videos.py
"""
Usage:
    python process_videos.py --input_dir ../data/raw --output_dir ../data/processed --side left
Requires: mediapipe, opencv-python, numpy, pandas
"""

import argparse
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import math
from tqdm import tqdm

mp_pose = mp.solutions.pose

def angle_deg(A, B, C):
    # A,B,C are (x,y) in pixels
    ux, uy = A[0]-B[0], A[1]-B[1]
    vx, vy = C[0]-B[0], C[1]-B[1]
    dot = ux*vx + uy*vy
    mu = math.hypot(ux, uy)
    mv = math.hypot(vx, vy)
    if mu == 0 or mv == 0:
        return float('nan')
    cos_theta = max(-1.0, min(1.0, dot/(mu*mv)))
    return math.degrees(math.acos(cos_theta))

def process_video(video_path, side='left', min_confidence=0.3):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    p = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.4, min_tracking_confidence=0.4)
    frame_idx = 0
    rows = []
    # choose landmark indices
    LM = mp_pose.PoseLandmark
    if side == 'left':
        hip_i, knee_i, ankle_i = LM.LEFT_HIP.value, LM.LEFT_KNEE.value, LM.LEFT_ANKLE.value
    else:
        hip_i, knee_i, ankle_i = LM.RIGHT_HIP.value, LM.RIGHT_KNEE.value, LM.RIGHT_ANKLE.value

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = p.process(image_rgb)
        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark
            hip = (lm[hip_i].x * w, lm[hip_i].y * h)
            knee = (lm[knee_i].x * w, lm[knee_i].y * h)
            ankle = (lm[ankle_i].x * w, lm[ankle_i].y * h)
            # average visibility as rough confidence
            conf = float(np.mean([lm[hip_i].visibility, lm[knee_i].visibility, lm[ankle_i].visibility]))
            angle = angle_deg(hip, knee, ankle)
        else:
            hip=knee=ankle=(np.nan,np.nan)
            conf = 0.0
            angle = np.nan
        rows.append({
            'frame': frame_idx,
            'timestamp_s': frame_idx / fps,
            'hip_x': hip[0], 'hip_y': hip[1],
            'knee_x': knee[0], 'knee_y': knee[1],
            'ankle_x': ankle[0], 'ankle_y': ankle[1],
            'angle_deg': angle,
            'confidence': conf
        })
        frame_idx += 1
    cap.release()
    p.close()
    return pd.DataFrame(rows)

def main(input_dir, output_dir, side):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_files = sorted(input_dir.glob("*.mp4"))
    for vf in video_files:
        print("Processing", vf.name)
        df = process_video(vf, side=side)
        outname = output_dir / (vf.stem + "_angles.csv")
        df.to_csv(outname, index=False)
        print("Saved:", outname)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--side", default="left", choices=['left','right'])
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.side)