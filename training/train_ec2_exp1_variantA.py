"""
Experiment 1 — Variant A: Cloud Preprocessing on EC2
Runs automatically 10 times and saves all results.

Pipeline per run:
  1. Download raw videos from S3 (only first run)
  2. Extract frames on EC2
  3. Run MediaPipe on EC2
  4. Train model on EC2

Usage:
  python3 train_ec2_exp1_variantA.py \
      --s3_bucket hiba-slr-filer \
      --s3_prefix exp1_raw_videos \
      --upload_time 88.95

Results saved to outputs/experiment1/
"""

import os
import time
import json
import argparse
import shutil

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from multiprocessing import Pool, cpu_count
from pathlib import Path

import mediapipe as mp
import boto3

from cnn_lstm import CNN_LSTM


# ===== SETTINGS =====

NUM_RUNS = 10
SEQUENCE_LENGTH = 16
IMG_SIZE = 224
BATCH_SIZE = 4
EPOCHS = 70
INSTANCE_TYPE = "g5.xlarge"
INSTANCE_PRICE_PER_HOUR = 1.006

VIDEO_DIR = "/tmp/exp1_raw_videos"
FRAMES_DIR = "/tmp/exp1_frames"
SKELETON_DIR = "/tmp/exp1_skeleton"
OUTPUT_DIR = "outputs/experiment1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ===== STEP 1: DOWNLOAD VIDEOS FROM S3 =====

def download_from_s3(bucket, s3_prefix, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=s3_prefix)
    total_files = 0
    t0 = time.perf_counter()
    for page in pages:
        for obj in page.get("Contents", []):
            s3_key = obj["Key"]
            rel_path = os.path.relpath(s3_key, s3_prefix)
            local_path = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(bucket, s3_key, local_path)
            total_files += 1
    elapsed = time.perf_counter() - t0
    print(f"Download complete: {total_files} files in {elapsed:.2f}s")
    return elapsed


# ===== STEP 2: EXTRACT FRAMES =====

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
                total_frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames_count == 0:
                    cap.release()
                    continue
                frame_indices = np.linspace(0, total_frames_count - 1, SEQUENCE_LENGTH).astype(int)
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


# ===== STEP 3: MEDIAPIPE =====

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
DrawingSpec = mp.solutions.drawing_utils.DrawingSpec
holistic_worker = None


def init_worker():
    global holistic_worker
    holistic_worker = mp_holistic.Holistic(
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
    results = holistic_worker.process(image_rgb)
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
    with Pool(cpu_count(), initializer=init_worker) as p:
        p.map(process_frame, tasks)
    elapsed = time.perf_counter() - t0
    print(f"MediaPipe: done in {elapsed:.2f}s")
    return elapsed


# ===== STEP 4: TRAINING =====

class VideoDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        for cls in classes:
            class_path = os.path.join(root_dir, cls)
            for video in os.listdir(class_path):
                video_path = os.path.join(class_path, video)
                if os.path.isdir(video_path):
                    self.samples.append((video_path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = sorted(os.listdir(video_path))
        if len(frames) < SEQUENCE_LENGTH:
            frames = frames + [frames[-1]] * (SEQUENCE_LENGTH - len(frames))
        else:
            frames = frames[:SEQUENCE_LENGTH]
        images = []
        for f in frames:
            img = Image.open(os.path.join(video_path, f)).convert("RGB")
            if self.transform:
                img = self.transform(img)
            images.append(img)
        return torch.stack(images), label


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def run_training(data_dir, run_id):
    train_loader = DataLoader(
        VideoDataset(os.path.join(data_dir, "train"), transform),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        VideoDataset(os.path.join(data_dir, "val"), transform),
        batch_size=BATCH_SIZE, num_workers=2, pin_memory=True
    )
    model = CNN_LSTM(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
    best_val_acc = 0
    t_start = time.perf_counter()
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for videos, labels in train_loader:
            videos, labels = videos.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                outputs = model(videos)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for videos, labels in val_loader:
                videos, labels = videos.to(device), labels.to(device)
                outputs = model(videos)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_model_{run_id}.pth")
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Val: {val_acc:.4f}")
    training_time = time.perf_counter() - t_start
    model.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_model_{run_id}.pth", map_location=device))
    test_loader = DataLoader(VideoDataset(os.path.join(data_dir, "test"), transform), batch_size=BATCH_SIZE)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for videos, labels in test_loader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = correct / total
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return training_time, best_val_acc, test_acc


# ===== MAIN =====

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3_bucket", default="hiba-slr-filer")
    parser.add_argument("--s3_prefix", default="exp1_raw_videos")
    parser.add_argument("--upload_time", type=float, required=True,
                        help="Upload time in seconds from stopwatch")
    parser.add_argument("--start_run", type=int, default=1,
                        help="Start from this run number (useful if resuming)")
    args = parser.parse_args()

    all_results = []

    for run_num in range(args.start_run, NUM_RUNS + 1):
        print(f"\n{'='*60}")
        print(f"=== Experiment 1 — Variant A — Run {run_num}/{NUM_RUNS} ===")
        print(f"{'='*60}\n")

        pipeline_start = time.perf_counter()

        # Step 1: Download from S3 (only first run)
        if run_num == args.start_run and not os.path.exists(VIDEO_DIR):
            download_time = download_from_s3(args.s3_bucket, args.s3_prefix, VIDEO_DIR)
        else:
            print("Skipping download (videos already local)")
            download_time = 0.0

        # Step 2: Extract frames (fresh each run)
        if os.path.exists(FRAMES_DIR):
            shutil.rmtree(FRAMES_DIR)
        frame_extraction_time = extract_frames(VIDEO_DIR, FRAMES_DIR)

        # Step 3: MediaPipe (fresh each run)
        if os.path.exists(SKELETON_DIR):
            shutil.rmtree(SKELETON_DIR)
        mediapipe_time = run_mediapipe(FRAMES_DIR, SKELETON_DIR)

        preprocessing_time = frame_extraction_time + mediapipe_time

        # Step 4: Train
        run_id = f"exp1_ec2_variantA_run{run_num}_{int(time.time())}"
        training_time, best_val_acc, test_acc = run_training(SKELETON_DIR, run_id)

        total_pipeline_time = time.perf_counter() - pipeline_start
        # Upload time only counted on run 1
        full_pipeline_time = (args.upload_time if run_num == 1 else 0) + total_pipeline_time
        cost_per_run = (full_pipeline_time / 3600) * INSTANCE_PRICE_PER_HOUR

        result = {
            "run": run_num,
            "run_id": run_id,
            "platform": "ec2",
            "variant": "A",
            "instance_type": INSTANCE_TYPE,
            "upload_time_sec": round(args.upload_time if run_num == 1 else 0, 4),
            "download_time_sec": round(download_time, 4),
            "frame_extraction_time_sec": round(frame_extraction_time, 4),
            "mediapipe_time_sec": round(mediapipe_time, 4),
            "preprocessing_time_sec": round(preprocessing_time, 4),
            "training_time_sec": round(training_time, 4),
            "total_pipeline_time_sec": round(full_pipeline_time, 4),
            "cost_per_run_usd": round(cost_per_run, 6),
            "best_val_accuracy": round(best_val_acc, 4),
            "test_accuracy": round(test_acc, 4)
        }

        # Save individual run result
        out_path = f"{OUTPUT_DIR}/exp1_ec2_variantA_run{run_num}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=4)

        all_results.append(result)

        print(f"\n--- Run {run_num} Results ---")
        print(f"Frame extraction:    {frame_extraction_time:.2f}s")
        print(f"MediaPipe:           {mediapipe_time:.2f}s")
        print(f"Training time:       {training_time:.2f}s")
        print(f"Total pipeline:      {full_pipeline_time:.2f}s")
        print(f"Cost:                ${cost_per_run:.4f}")
        print(f"Val accuracy:        {best_val_acc:.4f}")
        print(f"Test accuracy:       {test_acc:.4f}")

    # Save all runs summary
    summary_path = f"{OUTPUT_DIR}/exp1_ec2_variantA_all_runs.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"\n{'='*60}")
    print(f"All {NUM_RUNS} runs complete!")
    print(f"Individual results: {OUTPUT_DIR}/exp1_ec2_variantA_run*.json")
    print(f"All runs summary:   {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
