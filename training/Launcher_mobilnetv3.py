import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

boto_session = boto3.Session(region_name="eu-north-1")
sagemaker_session = sagemaker.Session(boto_session=boto_session)

estimator = PyTorch(
    entry_point="experiment3.py",
    source_dir=".",
    role="arn:aws:iam::600889066998:role/service-role/AmazonSageMaker-ExecutionRole-20260331T124966",
    instance_count=1,
    instance_type="ml.g5.xlarge",
    framework_version="2.0",
    py_version="py310",
    max_run=180000,
    sagemaker_session=sagemaker_session
)

for run in range(5):
    print(f"\n===== Launching SageMaker job {run+1}/5 (MobileNetV3) =====")
    estimator.fit({
        "train": "s3://hiba-slr-filer/data_skeleton/train",
        "val":   "s3://hiba-slr-filer/data_skeleton/val",
        "test":  "s3://hiba-slr-filer/data_skeleton/test"
    })
    print(f"Job {run+1} complete.")