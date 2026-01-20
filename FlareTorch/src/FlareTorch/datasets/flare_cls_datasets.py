import os
import hydra
import numpy as np
import pandas as pd

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.io import read_image

from loguru import logger as lgr_logger
from omegaconf import OmegaConf

from terratorch_surya.datasets.helio import HelioNetCDFDataset


class FlareHelioviewerClsDataset(Dataset):
    def __init__(
        self,
        task: str,
        index_path: str,
        input_time_delta: list[int],
        input_stat_path: str,
        limb_mask_path: str,
        scaler_mul: float,
        scaler_shift: float,
        scaler_div: float,
        label_type: str,
        phase: str,
    ):
        super().__init__()
        self.task = task
        self.input_time_delta = input_time_delta
        self.stats = OmegaConf.load(input_stat_path)  # input data statistics
        self.limb_mask = np.load(limb_mask_path)
        self.scaler_mul = scaler_mul
        self.scaler_shift = scaler_shift
        self.scaler_div = scaler_div
        self.label_type = label_type
        self.phase = phase

        # load index file
        self.index = pd.read_csv(index_path)
        self.index["timestamp"] = pd.to_datetime(self.index["timestamp"])
        self.index.set_index("timestamp", inplace=True)
        self.index.sort_index(inplace=True)
        self._get_valid_indices()
        lgr_logger.info(f"{self.phase} instances: {self.__len__()}")

        # Define Augmentation (Only for Training)
        if self.phase == "train":
            self.augment = transforms.Compose(
                [
                    transforms.RandomRotation(degrees=10),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                ]
            )
        else:
            self.augment = None  # No augmentation for validation/test

    def __len__(self):
        return len(self.valid_timestamps)

    def __getitem__(self, idx: int):
        current_time = self.valid_timestamps[idx]

        # Calculate all timestamps needed for this sample
        # e.g., current_time - 10min, current_time - 0min
        required_times = [
            current_time + pd.Timedelta(minutes=dt) for dt in self.input_time_delta
        ]

        # Load and stack all images
        images = []
        for t in required_times:
            # We know 't' exists because of _get_valid_indices validation
            img_path = self.index.loc[t, "input"]
            img = read_image(img_path)
            images.append(self.transform(img))

        # Stack along the first dimension (C, H, W) -> (Num_Frames, C, H, W)
        x = torch.stack(images, dim=1)
        x = x.float()

        target = self.mapping_target(self.index.loc[current_time, self.label_type])

        return x, torch.tensor(target, dtype=torch.float32), current_time.value

    def _get_valid_indices(self):
        time_deltas = pd.to_timedelta(self.input_time_delta, unit="min")
        idx = self.index.index

        valid_mask = np.ones(len(idx), dtype=bool)
        for dt in time_deltas:

            required_times = idx + dt
            has_required_time = required_times.isin(idx)
            valid_mask = valid_mask & has_required_time

        self.valid_timestamps = sorted(idx[valid_mask])

    def transform(self, data):

        data = data.float()

        # Apply Augmentation (Only if defined)
        if hasattr(self, "augment") and self.augment is not None:
            data = self.augment(data)

        data = data * self.limb_mask + 127.5 * (
            1 - self.limb_mask
        )  # limb regions become zero after the norm
        scaled = data * self.scaler_mul
        shift = scaled + self.scaler_shift

        return shift / self.scaler_div

    def mapping_target(self, target):

        if self.task == "regression" and self.label_type == "max_goes_class":

            target = target.strip().upper()

            if target.startswith("F"):
                return self.transform_target(1e-9)

            sub_class = float(target[1:])
            major_class = str(target[0])
            mapping_dict = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}

            base_flux = mapping_dict.get(major_class)
            if base_flux is None:
                raise ValueError(f"Unknown flare class: {major_class}")

            flux = sub_class * base_flux
            if flux <= 0.0:
                return self.transform_target(1e-9)
            return self.transform_target(flux)

    def transform_target(self, target):

        match self.task:
            case "regression":
                return np.log10(target) + 9


class FlareSuryaClsDataset(HelioNetCDFDataset):
    """
    The solar flare index data (flare_index_path) should be of the form

    timestamp,max_goes_class,cumulative_index,label_max,label_cum
    2011-01-01 00:00:00,B8.3,0.0,0,0
    2011-01-01 01:00:00,B8.3,0.0,0,0
    2011-01-01 02:00:00,B8.3,0.0,0,0
    2011-01-01 03:00:00,B8.3,0.0,0,0
    2011-01-01 04:00:00,B8.3,0.0,0,0
    2011-01-01 05:00:00,B8.3,0.0,0,0
    2011-01-01 06:00:00,B8.3,0.0,0,0
    """

    def __init__(
        self,
        sdo_data_root_path: str,
        index_path: str,
        flare_index_path: str,
        time_delta_input_minutes: list[int],
        time_delta_target_minutes: int,
        n_input_timestamps: int,
        rollout_steps: int,
        scalers=None,
        num_mask_aia_channels=0,
        drop_hmi_probability=0,
        use_latitude_in_learned_flow=False,
        channels: list[str] | None = None,
        phase="train",
        pooling: int | None = None,
        random_vert_flip: bool = False,
    ):

        self.flare_index = pd.read_csv(flare_index_path)
        self.flare_index["timestamp"] = pd.to_datetime(
            self.flare_index["timestamp"]
        ).values.astype("datetime64[ns]")
        self.flare_index.set_index("timestamp", inplace=True)
        self.flare_index.sort_index(inplace=True)

        super().__init__(
            sdo_data_root_path=sdo_data_root_path,
            index_path=index_path,
            time_delta_input_minutes=time_delta_input_minutes,
            time_delta_target_minutes=time_delta_target_minutes,
            n_input_timestamps=n_input_timestamps,
            rollout_steps=rollout_steps,
            scalers=scalers,
            num_mask_aia_channels=num_mask_aia_channels,
            drop_hmi_probability=drop_hmi_probability,
            use_latitude_in_learned_flow=use_latitude_in_learned_flow,
            channels=channels,
            phase=phase,
            pooling=pooling,
            random_vert_flip=random_vert_flip,
        )

        self.valid_indices = self.filter_valid_indices()
        self.adjusted_length = len(self.valid_indices)

    def filter_valid_indices(self) -> list:
        valid_indices = super().filter_valid_indices()

        valid_indices = [t for t in valid_indices if t in self.flare_index.index]

        return valid_indices

    def _get_index_data(self, idx: int) -> tuple[dict, dict]:
        data, metadata = super()._get_index_data(idx)

        reference_timestamp = self.valid_indices[idx]
        data["label"] = self.flare_index.loc[reference_timestamp, "label_max"]

        return data, metadata

    def __len__(self):
        return self.adjusted_length


@hydra.main(config_path="../../configs/", config_name="alexnet_helioviewer_config.yaml")
def main(cfg):
    dataset = FlareHelioviewerClsDataset(
        task=cfg.experiment.task,
        index_path=cfg.data.index_path.train,
        input_time_delta=cfg.data.input_time_delta,
        input_stat_path=cfg.data.input_stat_path,
        limb_mask_path=cfg.data.limb_mask_path,
        scaler_mul=cfg.data.scaler_mul,
        scaler_shift=cfg.data.scaler_shift,
        scaler_div=cfg.data.scaler_div,
        label_type=cfg.data.label_type,
        phase="training",
    )


if __name__ == "__main__":

    main()
