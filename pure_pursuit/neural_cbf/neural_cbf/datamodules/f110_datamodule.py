
from typing import List, Callable, Tuple, Dict, Optional

import torch
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader


from pure_pursuit.scripts.create_data import F110System, parse_args


class EpisodicDataModule(pl.LightningDataModule):
    def __init__(
        self,
        model: F110System,
        initial_domain: List[Tuple[float, float]],
        val_split: float = 0.1,
        batch_size: int = 64,
         quotas={"safe": 0.5, "unsafe": 0.3, "goal": 0.2},
    ):
        super().__init__()
        args = parse_args()
        self.model = model()
        self.initial_domain = initial_domain
        self.val_split = val_split
        self.batch_size = batch_size


    def prepare_data(self):


