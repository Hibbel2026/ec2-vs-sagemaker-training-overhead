"""
Experiment 1 — Variant B: Measure Local Preprocessing Time
Run this on your LOCAL desktop to measure how long
frame extraction + MediaPipe takes locally.

Usage:
  python3 measure_local_preprocessing.py \
      --video_dir ~/Desktop/asl_split \
      --frames_dir /tmp/local_frames \
      --skeleton_dir /tmp/local_skeleton

Result saved to: local_preprocessing_time.json
"""

import os
import time
import json
import argparse

import cv2
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count

import mediapipe as mp

SEQUENCE_LENGTH = 16

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
DrawingSpec = mp.solutions.drawing_utils.DrawingSpec

holistic = None


def init_worker():
    global holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )


def process_frame(args):
    input_path, output_path = args
    image = cv2.imread(str(input_path))
    if image is None:
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)
    left = DrawingSpec(color=(255, 0, 0), thickness=2)
    right = DrawingSpec(color=(0, 0, 255), thickness=2)
    pose = DrawingSpec(color=(0, 255, 255), thickness=2)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks,
                                  mp_holistic.POSE_CONNECTIONS, pose, pose)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                                  mp_holistic.HAND_CONNECTIONS, left, left)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                                  mp_holistic.HAND_CONNECTIONS, right, right)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def extract_frames(video_dir, output_dir):
    t0 = time.perf_counter()
    total_videos = 0
    for split in ["train", "val", "test"]:
        split_path = os.path.join(video_dir, split)
        if not os.path.isdir(split_path):
            continue
        for word in os.listdir(split_path):
            word_path = os.path.join(split_path, word)
            if not os.path.isdir(word_path):
                continue
            for video in os.listdir(word_path):
                if not video.endswith(".mp4"):
                    continue
                video_path = os.path.join(word_path, video)
                video_name = video.replace(".mp4", "")
                out_folder = os.path.join(output_dir, split, word, video_name)
                os.makedirs(out_folder, exist_ok=True)
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames == 0:
                    cap.release()
                    continue
                frame_indices = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH).astype(int)
                frame_id = 0
                saved_id = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_id in frame_indices:
                        filename = os.path.join(out_folder, f"frame_{saved_id:03d}.jpg")
                        cv2.imwrite(filename, frame)
                        saved_id += 1
                    frame_id += 1
                cap.release()
                total_videos += 1
    elapsed = time.perf_counter() - t0
    print(f"Frame extraction: {total_videos} videos in {elapsed:.2f}s")
    return elapsed


def run_mediapipe(frames_dir, skeleton_dir):
    input_root = Path(frames_dir)
    output_root = Path(skeleton_dir)
    tasks = []
    for frame in input_root.rglob("*.jpg"):
        rel = frame.relative_to(input_root)
        out = output_root / rel
        tasks.append((frame, out))
    print(f"MediaPipe: processing {len(tasks)} frames...")
    t0 = time.perf_counter()
    with Pool(4, initializer=init_worker) as p:
        p.map(process_frame, tasks)
    elapsed = time.perf_counter() - t0
    print(f"MediaPipe: done in {elapsed:.2f}s")
    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", default="~/Desktop/asl_split")
    parser.add_argument("--frames_dir", default="/tmp/local_frames")
    parser.add_argument("--skeleton_dir", default="/tmp/local_skeleton")
    args = parser.parse_args()

    video_dir = os.path.expanduser(args.video_dir)

    print("\n=== Measuring Local Preprocessing Time ===\n")

    frame_extraction_time = extract_frames(video_dir, args.frames_dir)
    mediapipe_time = run_mediapipe(args.frames_dir, args.skeleton_dir)
    total_preprocessing_time = frame_extraction_time + mediapipe_time

    result = {
        "platform": "local_desktop",
        "variant": "B",
        "frame_extraction_time_sec": round(frame_extraction_time, 4),
        "mediapipe_time_sec": round(mediapipe_time, 4),
        "total_preprocessing_time_sec": round(total_preprocessing_time, 4),
    }

    with open("local_preprocessing_time.json", "w") as f:
        json.dump(result, f, indent=4)

    print(f"\n=== Results ===")
    print(f"Frame extraction:   {frame_extraction_time:.2f}s")
    print(f"MediaPipe:          {mediapipe_time:.2f}s")
    print(f"Total preprocessing:{total_preprocessing_time:.2f}s")
    print(f"Saved to: local_preprocessing_time.json")


if __name__ == "__main__":
    main()
