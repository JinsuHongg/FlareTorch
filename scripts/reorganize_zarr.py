import zarr
from loguru import logger

def promote_arrays(zarr_path):
    """
    Promotes hmi_m and timestep arrays from a nested sub-group to the year group level.
    Structure change: /year/hmi_m/hmi_m -> /year/hmi_m
    """
    root = zarr.open(zarr_path, mode='a')
    years = sorted([k for k in root.group_keys() if k.isdigit()])
    
    for year in years:
        year_grp = root[year]
        # Check if the nested structure exists: year -> hmi_m (group) -> hmi_m (array)
        if 'hmi_m' in year_grp and isinstance(year_grp['hmi_m'], zarr.hierarchy.Group):
            logger.info(f"Promoting arrays for year {year}...")
            
            # 1. Rename the sub-group to avoid name collision with the target array name
            year_grp.move('hmi_m', 'tmp_move_src')
            
            # 2. Move the actual arrays up to the year level
            # These are metadata renames; chunks are not loaded into memory.
            if 'hmi_m' in year_grp['tmp_move_src']:
                year_grp.move('tmp_move_src/hmi_m', 'hmi_m')
            
            if 'timestep' in year_grp['tmp_move_src']:
                year_grp.move('tmp_move_src/timestep', 'timestep')
            
            # 3. Clean up the now-empty temporary group
            del year_grp['tmp_move_src']
            logger.success(f"Year {year} reorganized.")
        else:
            logger.info(f"Year {year} already has the correct structure or 'hmi_m' group is missing.")

if __name__ == "__main__":
    PATH = '/media/jhong90/storage/surya/sdo_512.zarr'
    promote_arrays(PATH)
