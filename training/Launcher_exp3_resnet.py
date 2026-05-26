"""
Experiment 3 — ResNet101: SageMaker Launcher
Starts SageMaker training jobs for the Large model in the scaling study.
ResNet101 + Bi-LSTM, trained on preprocessed frame sequences (Variant B pipeline).

Usage:
  python3 launcher_exp3_resnet101.py

Change NUM_RUNS to 5 when ready for full experiment.
Run a single iteration first (NUM_RUNS=1) to verify the job starts correctly
before launching the full sweep.
"""

import boto3
import sagemaker
import time
import json
import os
from sagemaker.pytorch import PyTorch


# ===== SETTINGS =====
NUM_RUNS = 5  # Experiment 3 uses 3–5 runs per model per platform


boto_session = boto3.Session(region_name="eu-north-1")
sagemaker_session = sagemaker.Session(boto_session=boto_session)


estimator = PyTorch(
    entry_point="train_sagemaker_resnet101.py",   # ← INTE train_ec2_resnet101.py!
    source_dir=".",
    role="arn:aws:iam::600889066998:role/service-role/AmazonSageMaker-ExecutionRole-20260331T124966",
    instance_count=1,
    instance_type="ml.g5.xlarge",
    framework_version="2.3",       # samma som MobileNetV3
    py_version="py311",            # samma som MobileNetV3
    max_run=180000,
    sagemaker_session=sagemaker_session
)


all_job_names = []

for run in range(1, NUM_RUNS + 1):

    print(f"\n{'='*60}")
    print(f"=== Launching SageMaker Job — ResNet101 — Run {run}/{NUM_RUNS} ===")
    print(f"{'='*60}\n")

    t0 = time.time()

    estimator.fit({
        "train": "s3://hiba-slr-filer/data_skeleton/train",
        "val":   "s3://hiba-slr-filer/data_skeleton/val",
        "test":  "s3://hiba-slr-filer/data_skeleton/test"
    })

    job_name = estimator.latest_training_job.name
    all_job_names.append(job_name)

    elapsed = time.time() - t0
    print(f"\nRun {run} complete. Job: {job_name} ({elapsed:.0f}s)")


print(f"\n{'='*60}")
print(f"All {NUM_RUNS} runs launched!")
print("Job names:", all_job_names)
print(f"{'='*60}")


# Save job names for reference
os.makedirs("outputs/experiment3", exist_ok=True)
with open("outputs/experiment3/exp3_sm_resnet101_jobs.json", "w") as f:
    json.dump({"jobs": all_job_names, "num_runs": NUM_RUNS, "model": "resnet101"}, f, indent=4)