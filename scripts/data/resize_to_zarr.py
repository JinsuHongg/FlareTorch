import argparse
import datetime
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import dask
import hdf5plugin  # Must be imported before xarray/h5netcdf to register compression plugins
import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
import zarr
from dask import delayed
from dask.distributed import Client, LocalCluster
from loguru import logger


def setup_logger(log_file: Path | None = None) -> None:
    """Configures loguru logger.

    Args:
        log_file: Optional path to save logs.
    """
    logger.remove()
    logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")
    if log_file:
        logger.add(log_file, format="{time} {level} {message}", level="DEBUG")


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Convert NetCDF to Zarr Tensor.")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory")
    parser.add_argument(
        "--output_zarr", type=str, required=True, help="Output Zarr path"
    )
    parser.add_argument(
        "--log_file", type=str, default="resize.log", help="Log file path"
    )
    parser.add_argument(
        "--target_size", type=int, default=512, help="Target image size"
    )
    parser.add_argument(
        "--batch_size", type=int, default=10, help="Files per Dask task"
    )
    parser.add_argument(
        "--scheduler", type=str, default=None, help="Dask scheduler address"
    )
    parser.add_argument(
        "--var_names",
        type=str,
        nargs="+",
        default=[
            "aia94",
            "aia131",
            "aia171",
            "aia193",
            "aia211",
            "aia304",
            "aia335",
            "aia1600",
            "hmi_m",
            "hmi_bx",
            "hmi_by",
            "hmi_bz",
            "hmi_v",
        ],
        help="List of variable names",
    )
    return parser.parse_args()


def extract_year(filename: str) -> str:
    """Extracts the year from the filename using a regex.

    Modify this function if your files have a specific naming convention
    that isn't caught by the simple 4-digit year regex (19XX or 20XX).

    Args:
        filename: Name of the file.

    Returns:
        String representing the year (e.g., '2011').
    """
    match = re.search(r"(19|20)\d{2}", filename)
    if match:
        return match.group(0)

    logger.warning(
        f"Could not extract year from {filename}. Defaulting to 'UNKNOWN_YEAR'."
    )
    return "UNKNOWN_YEAR"


def process_single_file(filepath: Path, var_names: list[str], target_size: int) -> Any:
    """Processes a single NetCDF file into a tensor.

    Args:
        filepath: Path to the NetCDF file.
        var_names: List of variable names to extract.
        target_size: Target height and width for resizing.

    Returns:
        Numpy array of shape (1, channels, H, W).
    """
    channels = []
    try:
        with xr.open_dataset(filepath, engine="h5netcdf") as ds:
            for var in var_names:
                if var in ds:
                    data = ds[var].values.astype(np.float32)
                    if data.ndim == 3:
                        data = data.squeeze()
                    data_tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
                    resized_tensor = F.interpolate(
                        data_tensor, size=(target_size, target_size), mode="area"
                    )
                    resized = resized_tensor.squeeze().numpy()
                    channels.append(resized)
                else:
                    channels.append(
                        np.zeros((target_size, target_size), dtype=np.float32)
                    )
        return np.stack(channels, axis=0)[np.newaxis, ...]
    except Exception as e:
        print(f"Failed to process {filepath}: {e}")
        return np.full(
            (1, len(var_names), target_size, target_size), np.nan, dtype=np.float32
        )


@delayed
def process_and_write_batch(
    filepaths: list[Path],
    var_names: list[str],
    target_size: int,
    start_idx: int,
    output_zarr_path: str,
    checkpoint_file: Path,
) -> bool:
    """Processes a batch of files and writes them directly to Zarr.

    Args:
        filepaths: List of paths to NetCDF files.
        var_names: Variables to extract.
        target_size: Target image size.
        start_idx: Starting index along the time (0th) axis for this batch.
        output_zarr_path: Path to the root Zarr store.
        checkpoint_file: Path to the checkpoint file to mark completion.

    Returns:
        True if successful, False otherwise.
    """
    try:
        if checkpoint_file.exists():
            return True

        batch_data = []
        batch_times = []
        for filepath in filepaths:
            tensor = process_single_file(filepath, var_names, target_size)
            batch_data.append(tensor)

            match = re.search(r"(\d{8}_\d{4})", filepath.name)
            if match:
                dt = datetime.datetime.strptime(match.group(1), "%Y%m%d_%H%M")
                ts_ns = int(dt.timestamp() * 1e9)
            else:
                ts_ns = 0
            batch_times.append(ts_ns)

        data_array = np.concatenate(batch_data, axis=0)
        time_array = np.array(batch_times, dtype=np.int64)

        # Write directly to Zarr Group
        root = zarr.open_group(output_zarr_path, mode="r+")
        end_idx = start_idx + len(filepaths)

        # The chunk boundaries line up with batch_size, ensuring no race conditions
        root["images"][start_idx:end_idx] = data_array  # type: ignore
        root["time"][start_idx:end_idx] = time_array  # type: ignore

        checkpoint_file.touch()
        return True
    except Exception as e:
        print(f"Error in batch starting at {start_idx}: {e}")
        return False


def main() -> None:
    """Main entrypoint for the conversion script."""
    args = parse_args()
    setup_logger(Path(args.log_file))

    input_dir = Path(args.input_dir)
    output_zarr = Path(args.output_zarr)
    checkpoints_dir = output_zarr / ".checkpoints"

    if args.scheduler:
        client = Client(args.scheduler)
        logger.info(f"Connected to Dask scheduler at {args.scheduler}")
    else:
        cluster = LocalCluster()
        client = Client(cluster)
        logger.info(f"Started local Dask cluster at {client.dashboard_link}")

    logger.info(f"Scanning {input_dir} for .nc files...")
    files = sorted(input_dir.rglob("*.nc"))
    if not files:
        logger.error("No .nc files found. Exiting.")
        return

    # Group files by year
    files_by_year: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        year = extract_year(f.name)
        files_by_year[year].append(f)

    logger.info(
        f"Found {len(files)} files spanning {len(files_by_year)} years: {list(files_by_year.keys())}"
    )

    channels = len(args.var_names)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    tasks = []

    # Pre-allocate Zarr store for each year
    for year, year_files in files_by_year.items():
        year_files = sorted(year_files)
        total_year_files = len(year_files)

        dataset_path = str(output_zarr / year / "dataset")

        if not Path(dataset_path).exists():
            logger.info(f"[{year}] Initializing new Zarr store at {dataset_path}")
            root = zarr.open_group(dataset_path, mode="w")
            
            img_arr = root.create_dataset(
                "images",
                shape=(total_year_files, channels, args.target_size, args.target_size),
                chunks=(args.batch_size, 1, args.target_size, args.target_size),
                dtype="float32",
            )
            img_arr.attrs["channel_names"] = args.var_names
            img_arr.attrs["_ARRAY_DIMENSIONS"] = ["time", "channel", "y", "x"]

            time_arr = root.create_dataset(
                "time",
                shape=(total_year_files,),
                chunks=(args.batch_size,),
                dtype="i8",
            )
            time_arr.attrs["_ARRAY_DIMENSIONS"] = ["time"]
            time_arr.attrs["units"] = "nanoseconds since 1970-01-01"
            time_arr.attrs["calendar"] = "proleptic_gregorian"
        else:
            logger.info(
                f"[{year}] Found existing Zarr store at {dataset_path}. Checking shape..."
            )
            root = zarr.open_group(dataset_path, mode="r+")
            if root["images"].shape[0] != total_year_files:  # type: ignore
                logger.error(
                    f"[{year}] Existing Zarr store has different total shape. Cannot resume properly!"
                )
                return

        logger.info(f"[{year}] Building task list for incomplete batches...")

        for i in range(0, total_year_files, args.batch_size):
            batch_idx = i // args.batch_size
            checkpoint_file = checkpoints_dir / f"batch_{year}_{batch_idx}.done"

            if checkpoint_file.exists():
                continue

            batch_files = year_files[i : i + args.batch_size]
            task = process_and_write_batch(
                batch_files,
                args.var_names,
                args.target_size,
                start_idx=i,
                output_zarr_path=dataset_path,
                checkpoint_file=checkpoint_file,
            )
            tasks.append(task)

    if not tasks:
        logger.info("All batches across all years are already completed!")
        return

    logger.info(f"Resuming/Starting: {len(tasks)} batches remaining. Executing...")

    # Compute all remaining delayed tasks
    results = dask.compute(*tasks)

    # Safely sum up results
    successes = sum([1 for r in results if r is True])
    logger.info(f"Finished. Successful batches: {successes}/{len(tasks)}")


if __name__ == "__main__":
    main()
