import os
import cv2
import yaml
import hydra
import numpy as np
# import pandas as pd
from loguru import logger as lgr_logger
import dask.array as da
from dask.delayed import delayed
from dask.diagnostics.progress import ProgressBar
from torchvision.io import read_image


def create_solar_limb_mask(image_path, threshold_value=10):
    """
    Creates a binary mask for the solar disk from a Helioviewer JPEG/PNG.
    
    Args:
        image_path (str): Path to the image file.
        threshold_value (int): Pixel value to separate Space (0) from Disk. 
                               10 is usually safe to catch the disk edge 
                               while ignoring compression noise.
    
    Returns:
        mask (numpy array): 0 for space, 1 for solar disk.
        center (tuple): (x, y) coordinates of the sun center.
        radius (float): Radius of the solar disk in pixels.
    """
    # Load Image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError("Could not load image. Check the path.")

    # Binary Threshold
    # Note: This might exclude dark sunspots initially, but we fix that next.
    _, binary_map = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)

    # Find Contours
    # This finds boundaries of all bright regions
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter for the Solar Disk
    # The sun is the largest object in the frame.
    if not contours:
        print("Warning: No contours found. Returning empty mask.")
        return np.zeros_like(img), (0,0), 0

    solar_disk_contour = max(contours, key=cv2.contourArea)

    # Fit a Circle
    (x, y), radius = cv2.minEnclosingCircle(solar_disk_contour)
    center = (int(x), int(y))
    radius = int(radius)
    
    # Optional: Slightly erode radius (e.g., 99%) to avoid limb darkening artifacts/noise
    # radius = int(radius * 0.99) 

    # Draw the Clean Mask
    mask = np.zeros_like(img)
    cv2.circle(mask, center, radius, (255), thickness=-1) # -1 fills the circle

    # Convert to boolean/binary (0 and 1)
    mask = mask // 255

    # save mask
    np.save('../../data/limb_mask.npy', mask)

    return mask, img, center, radius


def read_image_to_numpy(path):
    tensor = read_image(path) 
    return tensor.numpy()


def compute_stats(img_array, mask):
    lgr_logger.info(f"Computing statistics on array shape: {img_array.shape}")
    
    with ProgressBar():
        img_array = img_array.astype('float32')
        mask_expanded = mask[None, None, :, :]
        img_array = da.where(mask_expanded == 1, img_array, np.nan)

        min_val, max_val, mean_val, std_val = da.compute(
            da.nanmin(img_array),
            da.nanmax(img_array),
            da.nanmean(img_array),
            da.nanstd(img_array)
        )
        
        sum_val = da.nansum(img_array).compute()

    return {
        "min": float(min_val),
        "max": float(max_val),
        "sum": float(sum_val),
        "mean": float(mean_val),
        "std": float(std_val)
    }


def build_dask_stack(cfg):
    H, W = cfg.data.input.dim, cfg.data.input.dim
    ext = cfg.data.input.ext
    
    # Gather all file paths first
    file_paths = []
    for root, dirs, files in os.walk(cfg.data.input.path):
        for file in files:
            if file.endswith(ext):
                file_paths.append(os.path.join(root, file))
    
    if not file_paths:
        lgr_logger.warning("No files found!")
        return da.empty((0, H, W), dtype='float32')

    lgr_logger.info(f"Found {len(file_paths)} images.")

    # Create delayed objects
    lazy_imread = delayed(read_image_to_numpy)
    delayed_arrays = [lazy_imread(fp) for fp in file_paths]

    # Create Dask Array
    img_array = da.stack(
        [da.from_delayed(d, shape=(1, H, W), dtype='uint8') for d in delayed_arrays],
        axis=0
    )
    
    return img_array


@hydra.main(
    config_path="../../configs/", 
    config_name="alexnet_helioviewer_config.yaml",
    version_base=None
)
def main(cfg):

    if os.path.exists(cfg.data.limb_mask_path):
        lgr_logger.info(f"Mask array found at: {cfg.data.limb_mask_path}")
        mask = np.load(cfg.data.limb_mask_path)
    else:
        lgr_logger.info("Creating limb mask")
        # first create the solar limb mask
        ref_mask_img_path = os.path.join(
            cfg.data.input.path,
            "2010/12/06",
            cfg.data.ref_mask_img
            )
        mask, img, center, radius = create_solar_limb_mask(ref_mask_img_path)

    os.makedirs(os.path.dirname(cfg.data.input_stat_path), exist_ok=True)

    if os.path.exists(cfg.data.input_stat_path):
        lgr_logger.info(f"Statistics found at: {cfg.data.input_stat_path}")
    else:
        img_array = build_dask_stack(cfg)
        if img_array.shape[0] > 0:
            stats = compute_stats(img_array, mask)
            
            lgr_logger.info(f"Stats computed: {stats}")
            with open(cfg.data.input_stat_path, 'w') as file:
                yaml.dump(stats, file)
        else:
            lgr_logger.error("Image array is empty. No stats computed.")
    

if __name__ == "__main__":

    main()