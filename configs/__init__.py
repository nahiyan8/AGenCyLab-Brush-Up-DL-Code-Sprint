from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class Config:
    RANDOM_SEED: int
    PATH_DATASETS: str
    PATH_LOGS: str
    BATCH_SIZE: int
    NUM_WORKERS: int
    NUM_DEVICES: Optional[int]

from .default import default_cfg

cfg = default_cfg