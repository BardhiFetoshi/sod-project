import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")

RAW_IMG_DIR = os.path.join(DATA_DIR, "raw", "images")
RAW_MASK_DIR = os.path.join(DATA_DIR, "raw", "masks")

TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train", "images")
TRAIN_MASK_DIR = os.path.join(DATA_DIR, "train", "masks")

VAL_IMG_DIR = os.path.join(DATA_DIR, "val", "images")
VAL_MASK_DIR = os.path.join(DATA_DIR, "val", "masks")

TEST_IMG_DIR = os.path.join(DATA_DIR, "test", "images")
TEST_MASK_DIR = os.path.join(DATA_DIR, "test", "masks")

MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")
OVERLAYS_DIR = os.path.join(RESULTS_DIR, "overlays")

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model_20.pth")

IMAGE_SIZE = 224
BATCH_SIZE = 2
LEARNING_RATE = 0.001
EPOCHS = 20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(OVERLAYS_DIR, exist_ok=True)
