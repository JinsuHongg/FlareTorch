from pprint import pprint

import xarray as xr

if __name__ == "__main__":

    data_path  = "/media/jhong90/storage/surya/sdo_512.zarr/2010"

    # Open a Zarr dataset lazily
    ds = xr.open_zarr("/media/jhong90/storage/surya/sdo_512.zarr", group="2010/hmi_m", chunks="auto")

    # print all the variables and coords
    print(ds.variables)

    # Print Data variables
    print(ds.data_vars)

    # Print coordinates
    print(ds.coords)

    image = ds["hmi_m"].sel(timestep="2010-05-13T01:00:00").values
    pprint(image)



