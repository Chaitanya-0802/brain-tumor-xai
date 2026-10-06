from pathlib import Path

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]  # Fixed order, do not modify
IMG_SIZE = 128
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SEED = 42

# Colab & Drive paths
DRIVE_ROOT = Path("/content/drive/MyDrive/CSC507_BrainTumor")
DATA_DIR = Path("/content/data")  # Unzipped dataset location on Colab fast local disk
CKPT_DIR = DRIVE_ROOT / "checkpoints"
RESULTS_DIR = DRIVE_ROOT / "results"