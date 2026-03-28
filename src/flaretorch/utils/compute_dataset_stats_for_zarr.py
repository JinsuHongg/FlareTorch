import cv2
import yaml
import hydra
import numpy as np
import xarray as xr
import dask.array as da
from pathlib import Path
from dask.diagnostics.progress import ProgressBar
from loguru import logger as lgr_logger


def create_solar_limb_mask(image_path, threshold_value=10):
    """
    Creates a binary mask for the solar disk from a Helioviewer JPEG/PNG.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError("Could not load image. Check the path.")

    _, binary_map = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Warning: No contours found. Returning empty mask.")
        return np.zeros_like(img), img, (0, 0), 0

    solar_disk_contour = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(solar_disk_contour)
    center = (int(x), int(y))
    radius = int(radius)

    mask = np.zeros_like(img)
    cv2.circle(mask, center, radius, (255), thickness=-1)
    mask = mask // 255

    np.save(Path('../../data/limb_mask.npy'), mask)
    return mask, img, center, radius


def build_dask_stack_from_zarr(zarr_path, years=None, threshold_value=10):
    """
    Opens each year group from the zarr store, concatenates hmi_m along timestep,
    and creates a solar limb mask from the first valid image.
    
    Args:
        zarr_path (str | Path): Path to the root zarr store.
        years (list[int] | None): Years to load. Defaults to 2010–2024.
        threshold_value (int): Threshold for solar limb mask creation.
    
    Returns:
        full_stack (da.Array): shape (total_timesteps, 512, 512), dtype float32
        mask (np.ndarray): shape (H, W), values 0 or 1
    """
    if years is None:
        years = list(range(2010, 2025))

    zarr_path = Path(zarr_path)
    arrays = []
    mask = None

    for year in years:
        group_path = zarr_path / str(year)
        if not group_path.exists():
            lgr_logger.warning(f"Year group not found, skipping: {group_path}")
            continue

        lgr_logger.info(f"Opening year: {year}")
        ds = xr.open_zarr(group_path)

        if "hmi_m" not in ds:
            lgr_logger.warning(f"'hmi_m' not found in year {year}, skipping.")
            continue

        # Build limb mask from the first valid image in the first valid year
        if mask is None:
            lgr_logger.info(f"Creating limb mask from first image of year {year}...")
            first_image = ds["hmi_m"].isel(timestep=0).values  # (H, W), float32

            # Use absolute value — space is ~0 G, disk has signal regardless of polarity
            img_abs = np.abs(first_image)

            # Normalize by this image's own max (robust: no hardcoded range needed)
            img_uint8 = (img_abs / np.nanmax(img_abs) * 255).astype(np.uint8)

            # Otsu finds the disk/space boundary automatically from the histogram
            _, binary_map = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                raise RuntimeError("No contours found when creating limb mask. Try adjusting threshold_value.")

            solar_disk_contour = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(solar_disk_contour)
            center = (int(x), int(y))
            radius = int(radius)

            mask = np.zeros(first_image.shape, dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, thickness=-1)
            mask = mask // 255

            lgr_logger.info(f"Limb mask created — center: {center}, radius: {radius}px")

        arr = ds["hmi_m"].data.astype("float32")
        arrays.append(arr)
        lgr_logger.info(f"  → {year}: {arr.shape[0]} timesteps")

    if not arrays:
        raise RuntimeError("No valid year groups found in zarr store.")
    if mask is None:
        raise RuntimeError("Limb mask could not be created — no valid images found.")

    full_stack = da.concatenate(arrays, axis=0)
    lgr_logger.info(f"Total stack shape: {full_stack.shape}")

    return full_stack, mask


def compute_stats(img_array, mask):
    """
    Compute per-pixel statistics over the full time stack, inside the solar limb mask.
    
    Args:
        img_array: dask array, shape (T, H, W), float32
        mask: numpy array, shape (H, W), values 0 or 1
    
    Returns:
        dict with min, max, mean, std, sum
    """
    lgr_logger.info(f"Computing statistics on array shape: {img_array.shape}")

    with ProgressBar():
        mask_expanded = da.from_array(mask[np.newaxis, :, :], chunks=(1, mask.shape[0], mask.shape[1]))
        masked = da.where(mask_expanded == 1, img_array, np.nan)

        min_val, max_val, mean_val, std_val, sum_val = da.compute(
            da.nanmin(masked),
            da.nanmax(masked),
            da.nanmean(masked),
            da.nanstd(masked),
            da.nansum(masked),
        )

    return {
        "min":  float(min_val),
        "max":  float(max_val),
        "mean": float(mean_val),
        "std":  float(std_val),
        "sum":  float(sum_val),
    }


@hydra.main(
    config_path="../../configs/",
    config_name="alexnet_helioviewer_config.yaml",
    version_base=None,
)
def main(cfg):

    stat_path = Path(cfg.data.input_stat_path)
    stat_path.parent.mkdir(parents=True, exist_ok=True)

    if stat_path.exists():
        lgr_logger.info(f"Statistics found at: {stat_path}")
    else:
        img_array, mask = build_dask_stack_from_zarr(
            zarr_path=Path(cfg.data.zarr_path),
            years=list(range(2010, 2025)),
        )

        stats = compute_stats(img_array, mask)
        lgr_logger.info(f"Stats computed: {stats}")

        stat_path.write_text(yaml.dump(stats))


if __name__ == "__main__":
    main()
