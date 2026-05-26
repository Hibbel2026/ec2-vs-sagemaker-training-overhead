"""
Experiment 1 — Variant A: SageMaker Entry Point
This script runs ON the SageMaker instance.

Pipeline:
  1. Receive raw videos via SM_CHANNEL_RAWVIDEOS
  2. Extract frames on SageMaker instance
  3. Run MediaPipe on SageMaker instance
  4. Train model

Do NOT run this directly — use launcher_exp1_variantA.py
"""

import os
import time
import json
import argparse

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

from cnn_lstm import CNN_LSTM


# ===== SETTINGS =====

SEQUENCE_LENGTH = 16
IMG_SIZE = 224
BATCH_SIZE = 4
EPOCHS = 70
INSTANCE_TYPE = "ml.g5.xlarge"
INSTANCE_PRICE_PER_HOUR = 1.505

OUTPUT_DIR = os.environ.get("SM_MODEL_DIR", "outputs/experiment1")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FRAMES_DIR = "/tmp/exp1_frames"
SKELETON_DIR = "/tmp/exp1_skeleton"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ===== STEP 1: EXTRACT FRAMES =====

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


# ===== STEP 2: MEDIAPIPE =====

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


# ===== STEP 3: TRAINING =====

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


def run_training(train_dir, val_dir, test_dir, run_id):
    train_loader = DataLoader(
        VideoDataset(train_dir, transform),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        VideoDataset(val_dir, transform),
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
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"best_model_{run_id}.pth"))
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Val: {val_acc:.4f}")
    training_time = time.perf_counter() - t_start
    model.load_state_dict(torch.load(
        os.path.join(OUTPUT_DIR, f"best_model_{run_id}.pth"), map_location=device))
    test_loader = DataLoader(VideoDataset(test_dir, transform), batch_size=BATCH_SIZE)
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
    parser.add_argument("--upload_time", type=float, default=88.95)
    args = parser.parse_args()

    print("\n=== Experiment 1 — Variant A: Cloud Preprocessing on SageMaker ===\n")
    pipeline_start = time.perf_counter()

    # Raw videos downloaded by SageMaker into SM_CHANNEL_RAWVIDEOS
    raw_dir = os.environ["SM_CHANNEL_RAWVIDEOS"]
    print(f"Raw videos at: {raw_dir}")
    print(f"Contents: {os.listdir(raw_dir)[:5]}")

    # Step 1: Extract frames
    frame_extraction_time = extract_frames(raw_dir, FRAMES_DIR)

    # Step 2: MediaPipe
    mediapipe_time = run_mediapipe(FRAMES_DIR, SKELETON_DIR)

    preprocessing_time = frame_extraction_time + mediapipe_time

    # Step 3: Train
    run_id = f"exp1_sm_variantA_{int(time.time())}"
    training_time, best_val_acc, test_acc = run_training(
        os.path.join(SKELETON_DIR, "train"),
        os.path.join(SKELETON_DIR, "val"),
        os.path.join(SKELETON_DIR, "test"),
        run_id
    )

    total_pipeline_time = time.perf_counter() - pipeline_start
    full_pipeline_time = args.upload_time + total_pipeline_time
    cost_per_run = (full_pipeline_time / 3600) * INSTANCE_PRICE_PER_HOUR

    results = {
        "experiment": "experiment1_preprocessing_placement",
        "platform": "sagemaker",
        "variant": "A",
        "run_id": run_id,
        "instance_type": INSTANCE_TYPE,
        "upload_time_sec": round(args.upload_time, 4),
        "frame_extraction_time_sec": round(frame_extraction_time, 4),
        "mediapipe_time_sec": round(mediapipe_time, 4),
        "preprocessing_time_sec": round(preprocessing_time, 4),
        "training_time_sec": round(training_time, 4),
        "total_pipeline_time_sec": round(full_pipeline_time, 4),
        "cost_per_run_usd": round(cost_per_run, 6),
        "best_val_accuracy": round(best_val_acc, 4),
        "test_accuracy": round(test_acc, 4)
    }

    out_path = os.path.join(OUTPUT_DIR, f"exp1_sm_variantA_{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n=== Results — Variant A / SageMaker ===")
    print(f"Upload time:          {args.upload_time:.2f}s")
    print(f"Frame extraction:     {frame_extraction_time:.2f}s")
    print(f"MediaPipe:            {mediapipe_time:.2f}s")
    print(f"Preprocessing total:  {preprocessing_time:.2f}s")
    print(f"Training time:        {training_time:.2f}s")
    print(f"Total pipeline:       {full_pipeline_time:.2f}s")
    print(f"Cost per run:         ${cost_per_run:.4f}")
    print(f"Results saved to:     {out_path}")


if __name__ == "__main__":
    main()
