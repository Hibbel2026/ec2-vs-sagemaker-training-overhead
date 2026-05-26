"""
Experiment 1 — Variant A: Upload raw videos to S3
Run this on your LOCAL desktop BEFORE starting the EC2 job.

Usage:
  python3 upload_videos_local.py \
      --video_dir /path/to/asl_split \
      --s3_bucket hiba-slr-filer

Output:
  upload_result.json  (saved locally, contains upload_time_sec)
"""

import os
import time
import json
import argparse
import boto3


def upload_videos_to_s3(video_dir, bucket, s3_prefix):
    s3 = boto3.client("s3")
    total_files = 0
    total_bytes = 0

    t0 = time.perf_counter()

    for root, _, files in os.walk(video_dir):
        for fname in files:
            if not fname.endswith(".mp4"):
                continue
            local_path = os.path.join(root, fname)
            s3_key = s3_prefix + "/" + os.path.relpath(local_path, video_dir)
            s3.upload_file(local_path, bucket, s3_key)
            total_files += 1
            total_bytes += os.path.getsize(local_path)
            print(f"Uploaded: {s3_key}")

    elapsed = time.perf_counter() - t0
    total_gb = total_bytes / (1024 ** 3)

    print(f"\nUpload complete!")
    print(f"Files:     {total_files} videos")
    print(f"Size:      {total_gb:.2f} GB")
    print(f"Time:      {elapsed:.2f}s")

    return elapsed, total_files, total_gb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True,
                        help="Local path to asl_split (with train/val/test subdirs)")
    parser.add_argument("--s3_bucket", default="hiba-slr-filer")
    parser.add_argument("--s3_prefix", default="exp1_raw_videos")
    args = parser.parse_args()

    upload_time, total_files, total_gb = upload_videos_to_s3(
        args.video_dir, args.s3_bucket, args.s3_prefix
    )

    result = {
        "upload_time_sec": round(upload_time, 4),
        "total_files": total_files,
        "total_gb": round(total_gb, 4),
        "s3_bucket": args.s3_bucket,
        "s3_prefix": args.s3_prefix
    }

    with open("upload_result.json", "w") as f:
        json.dump(result, f, indent=4)

    print(f"\nSaved to: upload_result.json")
    print(f"Upload time: {upload_time:.2f}s — kopiera denna till ditt EC2 resultat!")


if __name__ == "__main__":
    main()
