"""
Experiment 1 — Variant A: SageMaker Launcher
Starts a SageMaker training job that downloads raw videos from S3,
runs frame extraction + MediaPipe on the instance, then trains.

Usage:
  python3 launcher_exp1_variantA.py

Change NUM_RUNS to 10 when ready for full experiment.
"""

import boto3
import sagemaker
import time
import json
from sagemaker.pytorch import PyTorch

# ===== SETTINGS =====
NUM_RUNS = 10  # Change to 10 for full experiment
UPLOAD_TIME = 88.95  # seconds — measured manually with stopwatch

boto_session = boto3.Session(region_name="eu-north-1")
sagemaker_session = sagemaker.Session(boto_session=boto_session)

estimator = PyTorch(
    entry_point="train_sagemaker_exp1_variantA.py",
    source_dir=".",
    role="arn:aws:iam::600889066998:role/service-role/AmazonSageMaker-ExecutionRole-20260331T124966",
    instance_count=1,
    instance_type="ml.g5.xlarge",
    framework_version="2.0",
    py_version="py310",
    max_run=180000,
    sagemaker_session=sagemaker_session,
    hyperparameters={
        "upload_time": UPLOAD_TIME,
    }
)

all_job_names = []

for run in range(1, NUM_RUNS + 1):
    print(f"\n{'='*60}")
    print(f"=== Launching SageMaker Job — Variant A — Run {run}/{NUM_RUNS} ===")
    print(f"{'='*60}\n")

    t0 = time.time()

    estimator.fit({
        "rawvideos": "s3://hiba-slr-filer/exp1_raw_videos"
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
with open("outputs/experiment1/exp1_sm_variantA_jobs.json", "w") as f:
    json.dump({"jobs": all_job_names, "num_runs": NUM_RUNS}, f, indent=4)
