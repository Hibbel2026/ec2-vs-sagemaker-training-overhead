"""
Experiment 3: ResNet101 + LSTM — SageMaker entry point

Runs inside a SageMaker training container. Reads data from SageMaker
channels (mounted automatically at /opt/ml/input/data/{train,val,test}),
trains ResNet101 + LSTM, and writes results to /opt/ml/model/ so
SageMaker uploads them to S3 on completion.

Launched by: Launcher_exp3_resnet.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import time
import json

from models import ResNet101LSTM

# ===== cuDNN settings =====
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# ===== SAGEMAKER PATHS =====
TRAIN_DIR  = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
VAL_DIR    = os.environ.get("SM_CHANNEL_VAL",   "/opt/ml/input/data/val")
TEST_DIR   = os.environ.get("SM_CHANNEL_TEST",  "/opt/ml/input/data/test")
MODEL_DIR  = os.environ.get("SM_MODEL_DIR",     "/opt/ml/model")
OUTPUT_DIR = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===== SETTINGS =====

IMG_SIZE                = 224
SEQUENCE_LENGTH         = 16
BATCH_SIZE              = 4
EPOCHS                  = 70
INSTANCE_TYPE           = "ml.g5.xlarge"
INSTANCE_PRICE_PER_HOUR = 1.505

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== HELPERS =====

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def sec_to_ms(seconds):
    return round(seconds * 1000, 4)


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
            img_path = os.path.join(video_path, f)
            try:
                img = Image.open(img_path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                images.append(img)
            except Exception as e:
                print(f"[WARN] Skipping {img_path}: {type(e).__name__}: {e}")
                images.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))
        return torch.stack(images), label

# ===== TRANSFORMS =====

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ===== DATA LOADERS =====

train_dataset = VideoDataset(TRAIN_DIR, transform)
val_dataset   = VideoDataset(VAL_DIR,   transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    num_workers=2,
    pin_memory=True
)


# ===== TRAINING =====

run_id = f"resnet101_sagemaker_{int(time.time())}"
print(f"\n===== Starting training run: {run_id} =====")

model      = ResNet101LSTM(num_classes=100).to(device)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {num_params:,}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scaler    = torch.amp.GradScaler(enabled=torch.cuda.is_available())

best_val_acc = 0
train_start  = time.perf_counter()
epoch_logs   = []

for epoch in range(EPOCHS):

    epoch_start = time.perf_counter()
    model.train()

    total_loss     = 0
    correct_train  = 0
    total_train    = 0

    epoch_data_loading = 0.0
    epoch_forward      = 0.0
    epoch_backward     = 0.0
    epoch_optimizer    = 0.0

    data_iter = iter(train_loader)

    for _ in range(len(train_loader)):

        t0 = time.perf_counter()
        videos, labels = next(data_iter)
        videos = videos.to(device)
        labels = labels.to(device)
        cuda_sync()
        epoch_data_loading += time.perf_counter() - t0

        optimizer.zero_grad()

        t1 = time.perf_counter()
        with torch.amp.autocast(
            device_type="cuda", enabled=torch.cuda.is_available()
        ):
            outputs = model(videos)
            loss    = criterion(outputs, labels)
        cuda_sync()
        epoch_forward += time.perf_counter() - t1

        _, predicted   = torch.max(outputs, 1)
        total_train   += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        t2 = time.perf_counter()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        cuda_sync()
        epoch_backward += time.perf_counter() - t2

        t3 = time.perf_counter()
        scaler.step(optimizer)
        scaler.update()
        cuda_sync()
        epoch_optimizer += time.perf_counter() - t3

        total_loss += loss.item()

    avg_loss  = total_loss / len(train_loader)
    train_acc = correct_train / total_train

    cuda_sync()
    epoch_wall_clock   = time.perf_counter() - epoch_start
    total_measured     = (epoch_data_loading + epoch_forward +
                          epoch_backward + epoch_optimizer)
    platform_overhead  = epoch_wall_clock - total_measured
    overhead_pct       = (platform_overhead / epoch_wall_clock) * 100
    cost_per_epoch_usd = (epoch_wall_clock / 3600) * INSTANCE_PRICE_PER_HOUR

    print(
        f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Wall: {epoch_wall_clock*1000:.1f}ms | "
        f"DL: {epoch_data_loading*1000:.1f}ms | "
        f"Fwd: {epoch_forward*1000:.1f}ms | "
        f"Bwd: {epoch_backward*1000:.1f}ms | "
        f"Opt: {epoch_optimizer*1000:.1f}ms | "
        f"Overhead: {platform_overhead*1000:.1f}ms ({overhead_pct:.1f}%)"
    )

    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for videos, labels in val_loader:
            videos  = videos.to(device)
            labels  = labels.to(device)
            outputs = model(videos)
            _, predicted = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total

    epoch_logs.append({
        "epoch":                    epoch + 1,
        "wall_clock_ms":            sec_to_ms(epoch_wall_clock),
        "data_loading_ms":          sec_to_ms(epoch_data_loading),
        "forward_pass_ms":          sec_to_ms(epoch_forward),
        "backward_pass_ms":         sec_to_ms(epoch_backward),
        "optimizer_step_ms":        sec_to_ms(epoch_optimizer),
        "total_measured_ms":        sec_to_ms(total_measured),
        "platform_overhead_ms":     sec_to_ms(platform_overhead),
        "overhead_percentage":      round(overhead_pct, 4),
        "cost_per_epoch_usd":       round(cost_per_epoch_usd, 8),
        "train_accuracy":           round(train_acc, 4),
        "val_accuracy":             round(val_acc, 4),
        "loss":                     round(avg_loss, 4)
    })

    print(f"Validation Accuracy: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(
            model.state_dict(),
            os.path.join(MODEL_DIR, f"best_model_{run_id}.pth")
        )
        print("Best model saved!")

print("Training finished")

train_total_time = time.perf_counter() - train_start
training_cost    = (train_total_time / 3600) * INSTANCE_PRICE_PER_HOUR

def mean(key): return sum(e[key] for e in epoch_logs) / len(epoch_logs)

avg_wall_clock     = mean("wall_clock_ms")
avg_data_loading   = mean("data_loading_ms")
avg_forward        = mean("forward_pass_ms")
avg_backward       = mean("backward_pass_ms")
avg_optimizer      = mean("optimizer_step_ms")
avg_total_measured = mean("total_measured_ms")
avg_overhead       = mean("platform_overhead_ms")
avg_overhead_pct   = mean("overhead_percentage")
avg_cost_per_epoch = mean("cost_per_epoch_usd")

print(f"Avg Epoch Wall Clock:   {avg_wall_clock:.1f} ms")
print(f"Avg Platform Overhead:  {avg_overhead:.1f} ms ({avg_overhead_pct:.1f}%)")
print(f"Total Training Time:    {train_total_time:.2f}s")
print(f"Training Cost:          ${training_cost:.4f}")

# ===== TEST =====
model.load_state_dict(
    torch.load(
        os.path.join(MODEL_DIR, f"best_model_{run_id}.pth"),
        map_location=device
    )
)

test_dataset = VideoDataset(TEST_DIR, transform)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model.eval()
correct = 0
total   = 0

with torch.no_grad():
    for videos, labels in test_loader:
        videos  = videos.to(device)
        labels  = labels.to(device)
        outputs = model(videos)
        _, predicted = torch.max(outputs, 1)
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()

test_acc = correct / total
print(f"TEST Accuracy: {test_acc:.4f}")

# ===== SAVE RESULTS =====
results = {
    "experiment":                   "experiment3_model_scaling",
    "platform":                     "sagemaker",
    "model_name":                   "resnet101",
    "model_parameters":             num_params,
    "data_source":                  "sagemaker_channels",
    "run_id":                       run_id,
    "instance_type":                INSTANCE_TYPE,
    "instance_price_per_hour_usd":  INSTANCE_PRICE_PER_HOUR,
    "batch_size":                   BATCH_SIZE,
    "epochs":                       EPOCHS,
    "sequence_length":              SEQUENCE_LENGTH,
    "train_time_sec":               round(train_total_time, 4),
    "training_cost_usd":            round(training_cost, 6),
    "avg_wall_clock_ms":            round(avg_wall_clock, 4),
    "avg_data_loading_ms":          round(avg_data_loading, 4),
    "avg_forward_pass_ms":          round(avg_forward, 4),
    "avg_backward_pass_ms":         round(avg_backward, 4),
    "avg_optimizer_step_ms":        round(avg_optimizer, 4),
    "avg_total_measured_ms":        round(avg_total_measured, 4),
    "avg_platform_overhead_ms":     round(avg_overhead, 4),
    "avg_overhead_percentage":      round(avg_overhead_pct, 4),
    "avg_cost_per_epoch_usd":       round(avg_cost_per_epoch, 8),
    "best_val_accuracy":            round(best_val_acc, 4),
    "test_accuracy":                round(test_acc, 4)
}

with open(os.path.join(MODEL_DIR, f"training_results_{run_id}.json"), "w") as f:
    json.dump(results, f, indent=4)

with open(os.path.join(MODEL_DIR, f"epoch_logs_{run_id}.json"), "w") as f:
    json.dump(epoch_logs, f, indent=4)

with open(os.path.join(OUTPUT_DIR, f"training_results_{run_id}.json"), "w") as f:
    json.dump(results, f, indent=4)

with open(os.path.join(OUTPUT_DIR, f"epoch_logs_{run_id}.json"), "w") as f:
    json.dump(epoch_logs, f, indent=4)

print(f"Results saved to {MODEL_DIR}/ and {OUTPUT_DIR}/")

if torch.cuda.is_available():
    torch.cuda.empty_cache()