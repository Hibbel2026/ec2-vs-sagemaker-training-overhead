"""
MediaPipe preprocessing using new API (mediapipe >= 0.10.30)
Compatible with Mac ARM64. Uses multiprocessing.

Usage:
  python3 mp_preprocess.py data/frames data_skeleton
"""

import argparse
import cv2
from pathlib import Path
from multiprocessing import Pool, cpu_count
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import urllib.request
import os


MODEL_DIR = Path.home() / ".mediapipe_models"
POSE_MODEL = str(MODEL_DIR / "pose_landmarker_lite.task")
HAND_MODEL = str(MODEL_DIR / "hand_landmarker.task")


def download_models():
    MODEL_DIR.mkdir(exist_ok=True)
    models = {
        POSE_MODEL: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        HAND_MODEL: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    }
    for path, url in models.items():
        if not os.path.exists(path):
            print(f"Downloading {os.path.basename(path)}...")
            urllib.request.urlretrieve(url, path)
            print(f"Downloaded.")


def init_worker():
    global pose_landmarker, hand_landmarker

    pose_options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL),
        running_mode=vision.RunningMode.IMAGE
    )
    pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

    hand_options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2
    )
    hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)


def process_frame(args):
    input_path, output_path = args

    image = cv2.imread(str(input_path))
    if image is None:
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    pose_result = pose_landmarker.detect(mp_image)
    hand_result = hand_landmarker.detect(mp_image)

    annotated = image.copy()

    if pose_result.pose_landmarks:
        for landmarks in pose_result.pose_landmarks:
            for lm in landmarks:
                h, w = image.shape[:2]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(annotated, (cx, cy), 3, (0, 255, 255), -1)

    if hand_result.hand_landmarks:
        for i, landmarks in enumerate(hand_result.hand_landmarks):
            handedness = hand_result.handedness[i][0].category_name
            color = (255, 0, 0) if handedness == "Left" else (0, 0, 255)
            for lm in landmarks:
                h, w = image.shape[:2]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(annotated, (cx, cy), 3, color, -1)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)


def process_dataset(input_root, output_root):
    input_root = Path(input_root)
    output_root = Path(output_root)

    tasks = []
    for frame in input_root.rglob("*.jpg"):
        rel = frame.relative_to(input_root)
        out = output_root / rel
        tasks.append((frame, out))

    print(f"Total frames: {len(tasks)}")
    print(f"Using 4 CPU cores")

    with Pool(4, initializer=init_worker) as p:
        p.map(process_frame, tasks)

    print("Done!")


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("input", default="data/frames", nargs="?")
    p.add_argument("output", default="data_skeleton", nargs="?")
    args = p.parse_args()

    download_models()
    process_dataset(args.input, args.output)


if __name__ == "__main__":
    _cli()
