from pathlib import Path

# ---------- Task definition (shared by everyone, never redefine elsewhere) ----------
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]  # fixed order
NUM_CLASSES = len(CLASSES)
IMG_SIZE = 128
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SEED = 42

# ---------- Paths (Colab) ----------
# Folder that DIRECTLY contains the "Training" and "Testing" folders
DATA_DIR = Path("/content/data")

DRIVE_ROOT = Path("/content/drive/MyDrive/CSC507_BrainTumor")
CKPT_DIR = DRIVE_ROOT / "checkpoints"
RESULTS_DIR = DRIVE_ROOT / "results"