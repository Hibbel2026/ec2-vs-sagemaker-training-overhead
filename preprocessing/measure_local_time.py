"""
Measure local preprocessing time for Experiment 1 Variant B.
Run this in your preprocessing folder:

  python3 measure_local_time.py

Make sure extract_frames.py and mp_preprocess.py are in the same folder.
"""

import time
import subprocess
import json
import os

print("\n=== Measuring Local Preprocessing Time ===\n")

# Step 1: Frame extraction
print("Step 1: Extracting frames...")
t0 = time.perf_counter()
subprocess.run(["python3", "extract_frames.py"], check=True)
t1 = time.perf_counter()
frame_extraction_time = t1 - t0
print(f"Frame extraction done: {frame_extraction_time:.2f}s\n")

# Step 2: MediaPipe
print("Step 2: Running MediaPipe...")
t2 = time.perf_counter()
subprocess.run(["python3", "mp_preprocess.py", "/tmp/local_frames", "/tmp/local_skeleton"], check=True)
t3 = time.perf_counter()
mediapipe_time = t3 - t2
print(f"MediaPipe done: {mediapipe_time:.2f}s\n")

total = frame_extraction_time + mediapipe_time

result = {
    "platform": "local_desktop",
    "variant": "B",
    "frame_extraction_time_sec": round(frame_extraction_time, 4),
    "mediapipe_time_sec": round(mediapipe_time, 4),
    "total_preprocessing_time_sec": round(total, 4)
}

with open("local_preprocessing_time.json", "w") as f:
    json.dump(result, f, indent=4)

print(f"=== Results ===")
print(f"Frame extraction:    {frame_extraction_time:.2f}s")
print(f"MediaPipe:           {mediapipe_time:.2f}s")
print(f"Total preprocessing: {total:.2f}s")
print(f"Saved to: local_preprocessing_time.json")
