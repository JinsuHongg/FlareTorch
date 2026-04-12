from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def query_fq(df, xray):
    
    timestamps = df[df["label_max"] == "FQ"].index # timestamps


    for timestamp in timestamps:
        xray["soft"].sel(timestep=timestamp).value.values.max()


if __name__ == "__main__":

    xray_zarr_path = "/scratch/users/jhong36/data"
    index_data_path = "/scratch/users/jhong36/data"
    time_delta = "24h"
    
    ds = xr.open_dataset(xray_zarr_path, chunks="auto")
    index = pd.read_csv(index_data_path)
    
    

