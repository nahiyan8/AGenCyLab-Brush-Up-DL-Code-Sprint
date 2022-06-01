from dataclasses import dataclass
from typing import Optional

from .default import default_cfg


@dataclass(frozen=True)
class Config:
    RANDOM_SEED: int
    PATH_DATASETS: str
    PATH_LOGS: str
    BATCH_SIZE: int
    NUM_WORKERS: int
    NUM_DEVICES: Optional[int]


cfg = default_cfg
