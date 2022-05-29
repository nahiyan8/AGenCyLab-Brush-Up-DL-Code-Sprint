import os
import torch

from . import Config

default_cfg = Config(
    RANDOM_SEED = 7,
    PATH_DATASETS = os.environ.get("PATH_DATASETS", "."),
    PATH_LOGS = os.environ.get("PATH_LOGS", "."),
    BATCH_SIZE = 256 if torch.cuda.is_available() else 64,
    NUM_WORKERS = int((os.cpu_count() or 2) / 2),
    NUM_DEVICES = 1 if torch.cuda.is_available() else None
)
