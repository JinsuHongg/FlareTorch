import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

from loguru import logger as lgr_logger
from omegaconf import OmegaConf


class FlareHelioviewerDataset(Dataset):
    def __init__(
        self,
        input_path: str,
        input_time_delta: list[int],
        input_stat_path: str,
        phase: str,
    ):
        super().__init__()
        self.input_time_delta = input_time_delta
        self.stats = OmegaConf.load(input_stat_path) # input data statistics
        self.phase = phase

        # load index file 
        self.index = pd.read_csv(input_path)
        self.index["timestamp"] = pd.to_datetime(self.index["timestamp"])
        self.index.set_index("timestamp", inplace=True)
        self.index.sort_index(inplace=True)
    
    def __len__(self):
        return len(self.valid_timestamps)

    def __getitem__(self, idx: int):
        timestamp = self.valid_timestamps[idx]
        return self.transform(read_image(self.index.loc[timestamp, "input_path"]))
        
    def _get_valid_indices(self):
        time_deltas = pd.to_timedelta(self.input_time_delta)
        idx = self.index.index

        # Start with the original index
        valid = set(idx)

        # Intersect with each shifted index
        for dt in time_deltas:
            shifted = set(idx - dt)
            valid &= shifted  # intersection

        self.valid_timestamps = sorted(valid)
    
    def transform(self, data):
        scaled = data * self.stats.scaler_factor
        log_data = np.log(scaled + self.stats.eps)

        return (log_data - self.stats.mean) / self.stats.std


if __name__ == "__main__":

    cfg = OmegaConf.load("../../configs/alexnet_helioviewer_config.yaml")

    dataset = FlareHelioviewerDataset(
        input_path=cfg.data.input_path,
        input_time_delta=cfg.data.input_time_delta,
        input_stat_path=cfg.data.input_stat_path,
        phase="training",
    )

