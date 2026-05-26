"""
Experiment 1: Preprocessing Placement (EC2)

Variant A: Upload raw frames to S3 → preprocess on EC2 instance → train
           Measures: upload_time, preprocessing_time, training_time, total_pipeline_time

Variant B: Load locally preprocessed frames directly → train
           Measures: training_time only (upload=0, preprocessing=0)

Usage:
  # Variant A (cloud preprocessing path):
  python train_ec2_exp1.py --variant A --raw_dir data/frames --s3_bucket hiba-slr-filer

  # Variant B (local preprocessing path):
  python train_ec2_exp1.py --variant B --preprocessed_dir data_skeleton

Run each variant 4 times for 8 total runs (Experiment 1 spec).
Results saved to outputs/experiment1/
"""

import argparse
import os
import time
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from cnn_lstm import CNN_LSTM


# ===== SETTINGS =====

IMG_SIZE = 224
SEQUENCE_LENGTH = 16
BATCH_SIZE = 4
EPOCHS = 70
INSTANCE_TYPE = "g5.xlarge"
INSTANCE_PRICE_PER_HOUR = 1.006

OUTPUT_DIR = "outputs/experiment1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== DATASET =====

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
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ===== PIPELINE STEPS =====

def upload_to_s3(local_dir, bucket, s3_prefix):
    """Upload a local directory tree to S3. Returns elapsed seconds."""
    import boto3
    s3 = boto3.client("s3")
    total_files = 0
    t0 = time.perf_counter()
    for root, _, files in os.walk(local_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            s3_key = s3_prefix + "/" + os.path.relpath(local_path, local_dir)
            s3.upload_file(local_path, bucket, s3_key)
            total_files += 1
    elapsed = time.perf_counter() - t0
    print(f"S3 upload: {total_files} files → s3://{bucket}/{s3_prefix} in {elapsed:.2f}s")
    return elapsed


def run_preprocessing(raw_dir, output_dir):
    """Apply MediaPipe holistic landmarks to all frames in raw_dir.
    Mirrors the logic in preprocessing/mp_preprocess.py.
    Returns elapsed seconds."""
    import mediapipe as mp
    import cv2

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    t0 = time.perf_counter()
    holistic = mp_holistic.Holistic(
        static_image_mode=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    processed = 0
    for root, _, files in os.walk(raw_dir):
        for fname in sorted(files):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            src = os.path.join(root, fname)
            rel = os.path.relpath(src, raw_dir)
            dst = os.path.join(output_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)

            img = cv2.imread(src)
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            cv2.imwrite(dst, img)
            processed += 1

    holistic.close()
    elapsed = time.perf_counter() - t0
    print(f"Preprocessing: {processed} frames in {elapsed:.2f}s")
    return elapsed


def run_training(data_dir, run_id):
    """Full training loop. Returns (training_time_sec, best_val_acc, test_acc)."""
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

    # Test
    model.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_model_{run_id}.pth", map_location=device))
    test_loader = DataLoader(
        VideoDataset(os.path.join(data_dir, "test"), transform), batch_size=BATCH_SIZE)
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
    parser = argparse.ArgumentParser(description="Experiment 1: Preprocessing Placement (EC2)")
    parser.add_argument("--variant", choices=["A", "B"], required=True,
                        help="A=cloud preprocess on EC2, B=use locally preprocessed frames")
    parser.add_argument("--raw_dir", default="data/frames",
                        help="[Variant A] Raw extracted frames directory (train/val/test subdirs)")
    parser.add_argument("--preprocessed_dir", default="data_skeleton",
                        help="[Variant B] Already preprocessed frames directory (train/val/test subdirs)")
    parser.add_argument("--s3_bucket", default="hiba-slr-filer",
                        help="[Variant A] S3 bucket for raw frame upload")
    parser.add_argument("--s3_prefix", default="exp1_raw_frames",
                        help="[Variant A] S3 key prefix")
    args = parser.parse_args()

    pipeline_start = time.perf_counter()
    upload_time = 0.0
    preprocessing_time = 0.0
    data_dir = args.preprocessed_dir

    if args.variant == "A":
        print("\n=== Variant A: Cloud Preprocessing on EC2 ===")
        upload_time = upload_to_s3(args.raw_dir, args.s3_bucket, args.s3_prefix)
        preprocess_output = "data_skeleton_exp1_variantA"
        os.makedirs(preprocess_output, exist_ok=True)
        preprocessing_time = run_preprocessing(args.raw_dir, preprocess_output)
        data_dir = preprocess_output
    else:
        print(f"\n=== Variant B: Local Preprocessing — loading from {data_dir} ===")

    run_id = f"exp1_ec2_variant{args.variant}_{int(time.time())}"
    training_time, best_val_acc, test_acc = run_training(data_dir, run_id)

    total_pipeline_time = time.perf_counter() - pipeline_start
    cost_per_run = (total_pipeline_time / 3600) * INSTANCE_PRICE_PER_HOUR

    results = {
        "experiment": "experiment1_preprocessing_placement",
        "platform": "ec2",
        "variant": args.variant,
        "run_id": run_id,
        "instance_type": INSTANCE_TYPE,
        "upload_time_sec": round(upload_time, 4),
        "preprocessing_time_sec": round(preprocessing_time, 4),
        "training_time_sec": round(training_time, 4),
        "total_pipeline_time_sec": round(total_pipeline_time, 4),
        "cost_per_run_usd": round(cost_per_run, 6),
        "best_val_accuracy": round(best_val_acc, 4),
        "test_accuracy": round(test_acc, 4)
    }

    out_path = f"{OUTPUT_DIR}/exp1_ec2_variant{args.variant}_{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n=== Results (Variant {args.variant} / EC2) ===")
    print(f"Upload time:       {upload_time:.2f}s")
    print(f"Preprocessing:     {preprocessing_time:.2f}s")
    print(f"Training time:     {training_time:.2f}s")
    print(f"Total pipeline:    {total_pipeline_time:.2f}s")
    print(f"Cost per run:      ${cost_per_run:.4f}")
    print(f"Results saved to:  {out_path}")


if __name__ == "__main__":
    main()
