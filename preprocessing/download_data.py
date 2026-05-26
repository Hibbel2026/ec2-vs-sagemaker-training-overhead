"""
=============================================================================
Sign Language Recognition - Dataset Download Instructions
=============================================================================
File: scripts/download_data.py
Description: Instructions for downloading the WLASL-100 dataset.

The dataset was downloaded manually from Kaggle:
https://www.kaggle.com/datasets/thtrnphc/wlasl100

Dataset location: data/preprocessing/
Structure:
    data/preprocessing/
    ├── train/frames/    (training images)
    ├── test/frames/     (test images)  
    └── val/frames/      (validation images)

Dataset info:
- Name: WLASL-100
- Size: ~625 MB
- Classes: 100 sign language words
- Format: JPG frames (pre-extracted from videos)
=============================================================================
"""

print("Dataset already downloaded!")
print("Location: data/preprocessing/")
print("Contains: train/, test/, val/ folders with JPG frames")