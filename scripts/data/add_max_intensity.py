import pandas as pd
import zarr
import numpy as np
from datetime import datetime, timedelta
import os
from loguru import logger
from collections import defaultdict

# Configuration
DATA_DIR = "/media/jhong90/storage/surya/index_data/"
ZARR_PATH = "/media/jhong90/storage/surya/xrs_24hour_slices_v2.zarr"
FILES = ["train.csv", "test.csv", "validation.csv", "leaky_validation.csv"]
EPOCH = datetime(2010, 4, 8, 0, 0, 0)

def goes_class_to_intensity(goes_class):
    """
    Converts GOES class string (e.g., 'B2.8', 'M1.5') to float intensity.
    Returns None for 'FQ' or invalid formats to trigger Zarr lookup.
    """
    if not isinstance(goes_class, str) or goes_class == 'FQ':
        return None
    
    goes_class = goes_class.strip().upper()
    if not goes_class:
        return None
        
    prefix = goes_class[0]
    try:
        value = float(goes_class[1:])
    except ValueError:
        return None
        
    multipliers = {
        'A': 1e-8,
        'B': 1e-7,
        'C': 1e-6,
        'M': 1e-5,
        'X': 1e-4
    }
    
    if prefix not in multipliers:
        return None
        
    return value * multipliers[prefix]

def process_index_files():
    if not os.path.exists(ZARR_PATH):
        logger.error(f"Zarr path not found: {ZARR_PATH}")
        return

    logger.info(f"Opening Zarr: {ZARR_PATH}")
    store = zarr.open(ZARR_PATH, mode='r')
    
    # timestep is in minutes since EPOCH (2010-04-08 00:00:00)
    logger.info("Loading timestep array...")
    timesteps = store['timestep'][:]
    xray = store['xray']
    
    # Pre-build a mapping from minutes_since_epoch to a list of Zarr indices
    ts_to_indices = defaultdict(list)
    for idx, ts in enumerate(timesteps):
        ts_to_indices[int(ts)].append(idx)
    
    logger.info(f"Loaded {len(ts_to_indices)} unique timestamps from Zarr.")
    
    # Cache for results to avoid re-reading Zarr for the same timestamp across files
    intensity_cache = {}

    for filename in FILES:
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            logger.warning(f"File {filepath} not found.")
            continue
            
        logger.info(f"Processing {filename}...")
        # Read the CSV
        df = pd.read_csv(filepath)
        
        # 1. Initialize max_intensity from GOES class string
        # This covers cases like A, B, C, M, X
        df['max_intensity'] = df['max_goes_class'].apply(goes_class_to_intensity)
        
        # 2. Fill in FQ or missing values from Zarr
        mask_needs_fill = df['max_intensity'].isna()
        if mask_needs_fill.any():
            num_to_fill = mask_needs_fill.sum()
            logger.info(f"Filling {num_to_fill} values from Zarr for {filename}...")
            
            # Helper to get intensity for a row
            def get_zarr_intensity(timestamp_str):
                try:
                    dt = pd.to_datetime(timestamp_str)
                except Exception:
                    return 0.0
                
                # t_zarr = t_csv + 24h (because Zarr slice [t-24, t] matches CSV [t, t+24])
                t_zarr = dt + timedelta(hours=24)
                minutes_since_epoch = int((t_zarr - EPOCH).total_seconds() / 60)
                
                if minutes_since_epoch in intensity_cache:
                    return intensity_cache[minutes_since_epoch]
                
                indices = ts_to_indices.get(minutes_since_epoch)
                if not indices:
                    # If not found exactly, we don't try to interpolate for now as per instructions
                    return 0.0 
                
                # Get max across all matching slices and the soft channel (index 0)
                max_val = 0.0
                for idx in indices:
                    # Access only the specific slice
                    slice_data = xray[idx, :, 0]
                    # np.nanmax handles possible NaNs in data. 
                    # If all elements are NaN, it may raise ValueError or return NaN with a warning.
                    try:
                        with np.errstate(invalid='ignore'):
                            current_max = np.nanmax(slice_data)
                    except (ValueError, RuntimeWarning): 
                        current_max = np.nan
                            
                    if not np.isnan(current_max):
                        max_val = max(max_val, float(current_max))
                
                intensity_cache[minutes_since_epoch] = max_val
                return max_val

            # Apply lookup only to rows that need it (FQ or invalid strings)
            # We use a progress update for large files
            df.loc[mask_needs_fill, 'max_intensity'] = df.loc[mask_needs_fill, 'timestamp'].apply(get_zarr_intensity)

        # Save the updated CSV
        df.to_csv(filepath, index=False)
        logger.success(f"Successfully updated {filename}")

if __name__ == "__main__":
    process_index_files()
