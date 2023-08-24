import torch

LR_RATE={"g": 3e-4, "d": 6e-4}
BATCH_SIZE=8
NUM_EPOCHS=1
NUM_WORKERS=4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"