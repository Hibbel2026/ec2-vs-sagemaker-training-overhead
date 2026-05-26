# Understanding Infrastructure Overhead in Cloud-Based Deep Learning Training: An Experimental Study of EC2 and SageMaker

## Project Description

This project compares Amazon EC2 (g5.xlarge) and AWS SageMaker (ml.g5.xlarge) for deep learning training overhead using sign language recognition as the workload. Both platforms run on identical GPU hardware (NVIDIA A10G) to isolate the effect of infrastructure management on training efficiency. The workload is a CNN+BiLSTM model trained on a 100-class American Sign Language (ASL) subset, and training time and cost are measured at the component level (data loading, forward pass, backward pass, optimizer step, and platform overhead) across three controlled experiments targeting preprocessing placement, overhead decomposition, and model complexity scaling.

---

## Repository Structure

```
Cloud-based-SLR-EC2-vs-Sagemaker/
├── data_exploration/
│   ├── build_top100_dataset.py        # Selects top 100 ASL words by video count
│   ├── split_dataset_balanced.py      # Splits into train/val/test (21/5/5 per class)
│   └── antal_ord.py                   # Counts words/videos in the raw dataset
│
├── preprocessing/
│   ├── extract_frames.py              # Extracts 16 evenly-spaced frames per video
│   ├── mp_preprocess.py               # Runs MediaPipe Holistic to annotate frames
│   ├── measure_local_preprocessing.py # Benchmarks local preprocessing time (4 CPU cores)
│   ├── measure_local_time.py          # Measures end-to-end local pipeline time
│   ├── upload_videos_local.py         # Uploads raw videos to S3
│   └── download_data.py               # Downloads data from S3
│
├── training/
│   │
│   │  -- Shared model definitions --
│   ├── cnn_lstm.py                    # ResNet50+BiLSTM baseline (Experiments 1 & 2)
│   ├── models.py                      # MobileNetV3LSTM, ResNet50LSTM, ResNet101LSTM (Experiment 3)
│   │
│   │  -- Experiment 1: Preprocessing Placement --
│   │  Compares uploading raw videos vs preprocessed frames;
│   │  measures upload + preprocessing + training time end-to-end.
│   ├── train_ec2_exp1.py              # EC2: Variant A (cloud preprocess) or B (local preprocess)
│   ├── train_sagemaker_exp1.py        # SageMaker: same variants via SM_CHANNEL_RAWVIDEOS
│   ├── launcher_exp1_variantA.py      # Launches SageMaker job for Experiment 1 Variant A
│   │
│   │  -- Experiment 2: Overhead Decomposition --
│   │  Measures per-component timing (data loading, forward, backward, optimizer)
│   │  and isolates platform overhead = wall clock − sum of components.
│   ├── train_ec2_exp2.py              # EC2: downloads data_skeleton from S3, 5 runs
│   ├── train_sagemaker_exp2.py        # SageMaker: reads from SM_CHANNEL_*, 1 run per job
│   ├── Launcher.py                    # Launches SageMaker job for Experiment 2
│   │
│   │  -- Experiment 3: Model Complexity Scaling --
│   │  Trains three models of increasing complexity to measure how overhead
│   │  scales with parameter count across EC2 and SageMaker.
│   ├── train_ec2_exp3_mobilenet.py    # EC2: MobileNetV3+BiLSTM (~7M params), local data
│   ├── train_ec2_exp3_resnet50.py     # EC2: ResNet50+BiLSTM (~30M params), local data
│   ├── train_ec2_exp3_resnet101.py    # EC2: ResNet101+BiLSTM (~49M params), local data
│   ├── train_sagemaker_exp3_mobilenet.py  # SageMaker: MobileNetV3+BiLSTM
│   ├── train_sagemaker_exp3_resnet50.py   # SageMaker: ResNet50+BiLSTM
│   ├── train_sagemaker_exp3_resnet101.py  # SageMaker: ResNet101+BiLSTM
│   ├── Launcher_mobilnetv3.py         # Launches SageMaker job for MobileNetV3
│   ├── Launcher_exp3_resnet.py        # Launches SageMaker job for ResNet50/101
│   │
│   │  -- Analysis --
│   ├── aggregate_sagemaker_exp2.py    # Aggregates multi-run SageMaker Experiment 2 results
│   └── plot_exp2_stacked_bar.py       # Plots stacked bar chart of component timing
│
├── outputs/
│   ├── experiment1/                   # JSON results for Experiment 1 runs
│   ├── experiment2/                   # JSON results for Experiment 2 runs
│   └── experiment3/                   # JSON results for Experiment 3 runs
│
└── requirements.txt
```

---

## Environment Setup

### EC2

- **OS**: Ubuntu 24.04
- **Python**: 3.12.3
- **PyTorch**: 2.3.0
- **CUDA**: 12.1

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### SageMaker

SageMaker jobs use an AWS-managed PyTorch container. No manual environment setup is required on the instance. The launcher scripts configure the container via:

```python
framework_version = "2.0"
py_version        = "py310"
```

All dependencies in `requirements.txt` should be listed under the `dependencies` argument of the `PyTorch` estimator or installed via a `requirements.txt` placed in the source directory.

---

## Dataset

1. Download from Hugging Face:
   [akasheroor/American-Sign-Language-Dataset](https://huggingface.co/datasets/akasheroor/American-Sign-Language-Dataset)

2. Select the top 100 words by video count:
   ```bash
   python data_exploration/build_top100_dataset.py
   ```

3. Split into train / val / test (21 / 5 / 5 videos per class):
   ```bash
   python data_exploration/split_dataset_balanced.py
   ```

4. Extract 16 evenly-spaced frames per video:
   ```bash
   python preprocessing/extract_frames.py
   ```

5. Run MediaPipe Holistic to annotate skeleton landmarks on each frame:
   ```bash
   python preprocessing/mp_preprocess.py
   ```

The preprocessed frames (annotated JPEGs) are stored in `data_skeleton/` and uploaded to S3 for use by EC2 Experiment 2 and all SageMaker experiments.

---

## How to Run

### Experiment 1 — Preprocessing Placement

**EC2 — Variant A** (preprocess raw frames on EC2, measure upload + preprocessing + training):
```bash
python training/train_ec2_exp1.py --variant A --raw_dir data/frames --s3_bucket hiba-slr-filer
```

**EC2 — Variant B** (use locally preprocessed frames, measure training only):
```bash
python training/train_ec2_exp1.py --variant B --preprocessed_dir data_skeleton
```

**SageMaker — Variant A** (upload raw videos to S3, preprocess on instance):
```bash
python training/launcher_exp1_variantA.py
```

Run each variant 4 times for a total of 8 runs per platform.

---

### Experiment 2 — Overhead Decomposition

**EC2** (downloads `data_skeleton` from S3 once, then runs 5 training runs):
```bash
python training/train_ec2_exp2.py
```

**SageMaker** (launch one job per run from your local machine; job reads data via SM channels):
```bash
python training/Launcher.py
```

Launch the SageMaker job 5 times to match the EC2 run count. Results are auto-saved to S3 by SageMaker; download and aggregate with:
```bash
python training/aggregate_sagemaker_exp2.py
```

---

### Experiment 3 — Model Complexity Scaling

**EC2** (reads from `data_skeleton/` locally, runs 5 times per model):
```bash
python training/train_ec2_exp3_mobilenet.py
python training/train_ec2_exp3_resnet50.py
python training/train_ec2_exp3_resnet101.py
```

**SageMaker** (launch one job per run per model):
```bash
python training/Launcher_mobilnetv3.py       # MobileNetV3
python training/Launcher_exp3_resnet.py      # ResNet50 or ResNet101 (set model_name in launcher)
```

---

## Results

EC2 consistently achieved lower wall-clock training time and cost per run than SageMaker across all three experiments, primarily due to SageMaker's managed-container overhead and per-component infrastructure costs. The overhead gap narrowed as model complexity increased — for ResNet101, SageMaker's relative overhead was smaller in percentage terms than for MobileNetV3 — suggesting that compute-bound workloads absorb infrastructure overhead more effectively. Experiment 1 showed that preprocessing placement has a measurable impact on total pipeline cost, with local preprocessing (Variant B) being more cost-efficient than cloud preprocessing (Variant A) on EC2. Refer to the full thesis for statistical analysis, per-epoch breakdowns, and cost projections.

---

## Author

Hibatallah Belhajali  
Master's Thesis in Computer Science  
KTH Royal Institute of Technology, 2026
