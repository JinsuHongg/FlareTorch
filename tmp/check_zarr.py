import xarray as xr

if __name__ == "__main__":

    data_path  = "/scratch/users/jhong36/data/sdo_512.zarr"

    # Open a Zarr dataset lazily
    ds = xr.open_zarr(data_path, chunks={})

    # Print Data variables
    print(ds.data_vars)

    # Print coordinates
    print(ds.coords)


