from sagemaker.pytorch import PyTorch
from sagemaker.debugger import ProfilerConfig, FrameworkProfile

profiler_config = ProfilerConfig(
    system_monitor_interval_millis=60000,
    framework_profile_params=FrameworkProfile()
)

estimator = PyTorch(
    entry_point="train_sagemaker_exp2.py",
    source_dir=".",          # krävs för att cnn_lstm.py ska inkluderas
    role="arn:aws:iam::600889066998:role/AmazonSageMaker-ExecutionRole-20260331T124966",
    instance_count=1,
    instance_type="ml.g5.xlarge",
    framework_version="2.0",
    py_version="py310",
    max_run=180000,
    profiler_config=profiler_config
)

for run in range(5):
    print(f"\n===== Launching SageMaker job {run+1}/5 =====")
    estimator.fit({
        "train": "s3://hiba-slr-filer/data_skeleton/train",
        "val":   "s3://hiba-slr-filer/data_skeleton/val",
        "test":  "s3://hiba-slr-filer/data_skeleton/test"
    })
    print(f"Job {run+1} complete.")