import dask.array as da
import hydra
import numpy as np
import pandas as pd
import xarray as xr
import zarr

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.io import read_image

from loguru import logger as lgr_logger
from omegaconf import OmegaConf

from terratorch_surya.datasets.helio import HelioNetCDFDataset


class FlareHelioviewerRegDataset(Dataset):
    """Dataset for solar flare regression using Helioviewer images.

    This dataset loads sequences of solar images from Helioviewer and pairs them
    with flare intensity labels for regression tasks.

    Args:
        input_index_path: Path to the CSV file containing image metadata.
        input_time_delta: List of time offsets (in minutes) for the input sequence.
        input_stat_path: Path to the YAML file containing data statistics.
        flare_index_path: Path to the CSV file containing flare labels.
        limb_mask_path: Path to the NPY file containing the solar limb mask.
        scaler_mul: Multiplicative factor for input scaling.
        scaler_shift: Additive shift for input scaling.
        scaler_div: Divisor for input scaling.
        label_type: Column name in flare_index for the target label.
        target_norm_type: Type of normalization for the target (e.g., 'log').
        phase: Dataset phase ('train', 'val', or 'test').

    Attributes:
        input_time_delta: List of time offsets for the input sequence.
        stats: Loaded data statistics.
        limb_mask: Solar limb mask array.
        scaler_mul: Multiplicative factor for input scaling.
        scaler_shift: Additive shift for input scaling.
        scaler_div: Divisor for input scaling.
        label_type: Column name for the target label.
        target_norm_type: Type of normalization for the target.
        phase: Dataset phase.
        index: Image metadata index.
        flare_index: Flare labels index.
        augment: Augmentation pipeline for training.
        valid_timestamps: List of timestamps with valid input sequences and labels.
    """

    def __init__(
        self,
        input_index_path: str,
        input_time_delta: list[int],
        input_stat_path: str,
        flare_index_path: str,
        limb_mask_path: str,
        scaler_mul: float,
        scaler_shift: float,
        scaler_div: float,
        label_type: str,
        target_norm_type: str,
        phase: str,
    ):
        super().__init__()
        self.input_time_delta = input_time_delta
        self.stats = OmegaConf.load(input_stat_path)  # input data statistics
        self.limb_mask = np.load(limb_mask_path)
        self.scaler_mul = scaler_mul
        self.scaler_shift = scaler_shift
        self.scaler_div = scaler_div
        self.label_type = label_type
        self.target_norm_type = target_norm_type
        self.phase = phase

        # load index file
        self.index = pd.read_csv(input_index_path)
        self.index["timestamp"] = pd.to_datetime(self.index["timestamp"])
        self.index.set_index("timestamp", inplace=True)
        self.index.sort_index(inplace=True)
        self.flare_index = pd.read_csv(flare_index_path)
        self.flare_index["timestamp"] = pd.to_datetime(self.flare_index["timestamp"])
        self.flare_index.set_index("timestamp", inplace=True)
        self.flare_index.sort_index(inplace=True)
        self._get_valid_indices()
        lgr_logger.info(f"{self.phase} instances: {self.__len__()}")

        # Define Augmentation (Only for Training)
        if self.phase == "train":
            self.augment = transforms.Compose(
                [
                    transforms.RandomRotation(degrees=11),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                ]
            )
        else:
            self.augment = None  # No augmentation for validation/test

    def __len__(self):
        """Returns the number of samples in the dataset.

        Returns:
            The number of valid timestamps.
        """
        return len(self.valid_timestamps)

    def __getitem__(self, idx: int):
        """Returns a single sample from the dataset.

        Args:
            idx: Index of the sample.

        Returns:
            A tuple containing (input_tensor, target_tensor, timestamp).
        """
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

        target = self.transform_target(
            self.flare_index.loc[current_time, self.label_type]
        )

        return x, torch.tensor(target, dtype=torch.float32), current_time.value

    def _get_valid_indices(self):
        time_deltas = pd.to_timedelta(self.input_time_delta, unit="min")
        idx = self.index.index

        valid_mask = np.ones(len(idx), dtype=bool)
        for dt in time_deltas:
            required_times = idx + dt
            has_required_time = required_times.isin(idx)
            valid_mask = valid_mask & has_required_time

        # Get timestamps that have valid input sequences
        valid_sequence_timestamps = idx[valid_mask]

        # Keep only timestamps that are ALSO in flare_index
        # This assumes both indices are DatetimeIndex
        final_valid_timestamps = valid_sequence_timestamps.intersection(
            self.flare_index.index
        )

        self.valid_timestamps = sorted(final_valid_timestamps)

    def transform(self, data):
        """Applies transformations and scaling to the input image.

        Args:
            data: Input image tensor.

        Returns:
            Transformed and scaled image tensor.
        """
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

    def transform_target(self, target):
        """Applies normalization to the target label.

        Args:
            target: Raw target value.

        Returns:
            Normalized target value.
        """
        match self.target_norm_type:
            case "log":
                if target == 0:
                    print("target is zero?")
                return np.log10(target) + 9


class FlareSuryaBenchDataset(Dataset):
    """Dataset for solar flare regression/classification using surya-bench data.

    Loads sequences of HMI magnetogram images from a Zarr store and pairs them
    with flare intensity labels for regression or binary classification tasks.

    Args:
        input_zarr_path: Path to the root Zarr store containing year groups.
        input_time_delta: List of time offsets (in minutes) for the input sequence.
            e.g. [-10, 0] loads two frames: 10 minutes before and at current_time.
        input_stat_path: Path to the YAML file containing dataset mean and std statistics.
        flare_index_path: Path to the CSV file containing flare labels with a
            'timestamp' column and one or more label columns.
        limb_mask_path: Path to the NPY file containing the solar limb mask (512x512).
        scaler_mul: Multiplicative scaler applied during preprocessing.
        scaler_shift: Shift scaler applied during preprocessing.
        scaler_div: Divisive scaler applied during preprocessing.
        label_type: Column name in flare_index for the target label.
        target_norm_type: Normalization strategy for the target label.
            'log'    — log10(target) + 9, returns float.
            'binary' — casts existing 0/1 column to int, returns long.
        phase: Dataset phase, one of 'train', 'val', or 'test'.
            Training phase applies random augmentations.

    Attributes:
        years: Sorted list of year group strings found in the Zarr store.
        index_timestamps: Flat DatetimeIndex of all available image timestamps.
        input_time_delta: List of time offsets (minutes) for the input sequence.
        stats: Loaded data statistics (mean, std) from the YAML file.
        limb_mask: Solar limb mask array of shape (512, 512).
        label_type: Column name for the target label.
        target_norm_type: Normalization strategy for the target label.
        phase: Dataset phase ('train', 'val', or 'test').
        flare_index: DataFrame of flare labels indexed by timestamp.
        augment: Augmentation pipeline for training, or None for val/test.
        valid_timestamps: Sorted list of timestamps that have both a complete
            input sequence and a matching entry in flare_index.
    """

    def __init__(
        self,
        input_zarr_path: str,
        input_time_delta: list[int],
        input_stat_path: str,
        flare_index_path: str,
        limb_mask_path: str,
        label_type: str,
        target_norm_type: str,
        phase: str,
    ):
        super().__init__()

        # Find year groups from Zarr store
        root = zarr.open(input_zarr_path, mode="r")
        self.years = sorted(root.group_keys(), key=int)

        # Open each year group as a lazy xarray DataArray
        self._arrays: dict[str, xr.DataArray] = {}
        for year in self.years:
            ds = xr.open_zarr(input_zarr_path, group=year)
            self._arrays[year] = ds["hmi_m"]  # DataArray: (timestep, x, y)

        # Build flat index — single concatenated DatetimeIndex
        index_timestamps = []
        for da in self._arrays.values():
            index_timestamps.append(pd.DatetimeIndex(da.coords["timestep"].values))
        self.index_timestamps = pd.DatetimeIndex(np.concatenate(index_timestamps))

        self.input_time_delta = input_time_delta
        self.stats = OmegaConf.load(input_stat_path)
        self.limb_mask = np.load(limb_mask_path)
        self.label_type = label_type
        self.target_norm_type = target_norm_type
        self.phase = phase

        self.flare_index = pd.read_csv(flare_index_path)
        self.flare_index["timestamp"] = pd.to_datetime(self.flare_index["timestamp"])
        self.flare_index.set_index("timestamp", inplace=True)
        self.flare_index.sort_index(inplace=True)

        self._get_valid_indices()
        lgr_logger.info(f"{self.phase} instances: {self.__len__()}")

        # Augmentation for training only
        if self.phase == "train":
            self.augment = transforms.Compose(
                [
                    transforms.RandomRotation(degrees=11),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                ]
            )
        else:
            self.augment = None

    def get_by_timestamp(self, timestamp: str | pd.Timestamp) -> np.ndarray:
        """Fetch a single HMI image by its exact timestamp.

        Args:
            timestamp: Timestamp string (e.g. '2015-06-21 12:00:00') or pd.Timestamp.

        Returns:
            Image array of shape (512, 512) as float32.

        Raises:
            KeyError: If the year of the timestamp is not loaded in this dataset.
        """
        ts = pd.Timestamp(timestamp)
        year = str(ts.year)
        if year not in self._arrays:
            raise KeyError(f"Year {year} not loaded in this dataset.")
        image = self._arrays[year].sel(timestep=ts).values.astype(np.float32)
        return image

    def __len__(self) -> int:
        """Returns the number of valid samples in the dataset.

        Returns:
            Number of valid timestamps.
        """
        return len(self.valid_timestamps)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Returns a single sample from the dataset.

        Args:
            idx: Index of the sample.

        Returns:
            Tuple of:
                x       — input tensor of shape (1, num_frames, 512, 512).
                target  — scalar target tensor (float32 for 'log', long for 'binary').
                timestamp — current_time as int64 (nanoseconds since epoch).
        """
        current_time = self.valid_timestamps[idx]

        required_times = [
            current_time + pd.Timedelta(minutes=dt) for dt in self.input_time_delta
        ]

        images = []
        for t in required_times:
            img = self.get_by_timestamp(t)
            images.append(self.transform(img))

        # (1, 512, 512) per frame → stack → (1, num_frames, 512, 512)
        x = torch.stack(images, dim=1).float()

        raw_target = self.flare_index.loc[current_time, self.label_type]
        target = self.transform_target(raw_target)

        dtype = torch.long if self.target_norm_type == "binary" else torch.float32
        return x, torch.tensor(target, dtype=dtype), current_time.value

    def _get_valid_indices(self) -> None:
        """Builds self.valid_timestamps by filtering index_timestamps to only those
        where every required input frame exists and a flare label is available.
        """
        time_deltas = pd.to_timedelta(self.input_time_delta, unit="min")
        idx = self.index_timestamps

        valid_mask = np.ones(len(idx), dtype=bool)
        for dt in time_deltas:
            required_times = idx + dt
            has_required_time = required_times.isin(idx)
            valid_mask = valid_mask & has_required_time

        valid_sequence_timestamps = idx[valid_mask]
        final_valid_timestamps = valid_sequence_timestamps.intersection(
            self.flare_index.index
        )
        self.valid_timestamps = sorted(final_valid_timestamps)

    def transform(self, arr: np.ndarray) -> torch.Tensor:
        """Applies signed log1p transform and standardization to an input image.

        The signed log1p transform compresses the dynamic range of magnetogram
        values while preserving the sign: f(x) = sign(x) * log1p(|x|).
        The result is then standardized using the precomputed dataset mean and std.

        Args:
            arr: Input image array of shape (512, 512), float32.

        Returns:
            Transformed image tensor of shape (1, 512, 512).
        """
        arr_transformed = np.sign(arr) * np.log1p(np.abs(arr))
        arr_normalized = (arr_transformed - self.stats.mean) / self.stats.std
        return torch.from_numpy(arr_normalized).unsqueeze(0)

    def transform_target(self, target: float) -> float | int:
        """Applies normalization to the target label.

        Args:
            target: Raw target value from flare_index.

        Returns:
            For 'log'    — float: log10(target) + 9.
            For 'binary' — int: 0 or 1 (cast from existing binary column).

        Raises:
            ValueError: If target_norm_type is not 'log' or 'binary'.
            ValueError: If target_norm_type is 'log' and target is zero.
        """
        match self.target_norm_type:
            case "log":
                if target == 0:
                    raise ValueError(
                        f"target is zero at — log10(0) is undefined. "
                        f"Check your flare_index for label_type='{self.label_type}'."
                    )
                return np.log10(target) + 9

            case "binary":
                return int(target)

            case _:
                raise ValueError(
                    f"Unknown target_norm_type: '{self.target_norm_type}'. "
                    f"Expected 'log' or 'binary'."
                )


class FlareSuryaClsDataset(HelioNetCDFDataset):
    """Dataset for solar flare classification using Surya/SDO data.

    This dataset extends HelioNetCDFDataset to include flare classification
    labels from a flare index file.

    Args:
        sdo_data_root_path: Root directory for SDO data.
        index_path: Path to the NetCDF index file.
        flare_index_path: Path to the CSV file containing flare labels.
        time_delta_input_minutes: List of time offsets for input.
        time_delta_target_minutes: Time offset for the target.
        n_input_timestamps: Number of input timestamps.
        rollout_steps: Number of rollout steps.
        scalers: Scalers for data normalization.
        num_mask_aia_channels: Number of AIA channels to mask.
        drop_hmi_probability: Probability of dropping HMI data.
        use_latitude_in_learned_flow: Whether to use latitude in learned flow.
        channels: List of channels to use.
        phase: Dataset phase ('train', 'val', or 'test').
        pooling: Pooling factor.
        random_vert_flip: Whether to apply random vertical flip.

    Attributes:
        flare_index: Flare labels index.
        valid_indices: List of valid timestamps.
        adjusted_length: Number of valid samples.
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
        """Filters timestamps to include only those present in the flare index.

        Returns:
            List of valid timestamps.
        """
        valid_indices = super().filter_valid_indices()

        valid_indices = [t for t in valid_indices if t in self.flare_index.index]

        return valid_indices

    def __len__(self):
        """Returns the number of samples in the dataset.

        Returns:
            The adjusted length of the dataset.
        """
        return self.adjusted_length


@hydra.main(config_path="../../configs/", config_name="alexnet_helioviewer_config.yaml")
def main(cfg):
    dataset = FlareHelioviewerRegDataset(
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
