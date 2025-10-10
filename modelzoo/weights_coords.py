# meps_weights.py
import os
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point
from pyproj import CRS

DEFAULT_DS_URL = "https://thredds.met.no/thredds/dodsC/mepslatest/archive/meps_det_2_5km_latest.nc"
DEFAULT_ROTATED_PROJ4 = "+proj=lcc +lat_0=63.3 +lon_0=15 +lat_1=63.3 +lat_2=63.3 +no_defs +R=6371000.0" 

# ---------- helpers ----------

def refine_grid(x, y, factor=5, buffer_m=1200):
    """
    Refine a grid by 'factor' and add +/- buffer_m on each axis before refinement.
    Returns 2D arrays (x_grid, y_grid).
    """
    x_min, x_max = x[0] - buffer_m, x[-1] + buffer_m
    y_min, y_max = y[0] - buffer_m, y[-1] + buffer_m
    x_new = np.linspace(x_min, x_max, len(x) * factor)
    y_new = np.linspace(y_min, y_max, len(y) * factor)
    return np.meshgrid(x_new, y_new)

def subset_dataset(ds: xr.Dataset, shape: gpd.GeoDataFrame, buffer_m=2000) -> xr.Dataset:
    """
    Subset ds by the shape's bounding box with +/- buffer_m (same units as ds x,y).
    """
    minx, miny, maxx, maxy = shape.total_bounds
    return ds.sel(
        x=slice(minx - buffer_m, maxx + buffer_m),
        y=slice(miny - buffer_m, maxy + buffer_m),
    )

def load_and_project_shape(shapefile_path: str, rotated_proj4: str) -> gpd.GeoDataFrame:
    """
    Load a shapefile and reproject it to the rotated grid CRS.
    If missing CRS, assume EPSG:4326 (change if needed).
    """
    rotated_crs = CRS.from_proj4(rotated_proj4)
    gdf = gpd.read_file(shapefile_path)
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)  # adjust if your data uses another CRS
    return gdf.to_crs(rotated_crs)

# ---------- core ----------
def compute_coords_weights(
    shapefile_path: str,
    ds: xr.Dataset | None = None,
    ds_url: str = DEFAULT_DS_URL,
    rotated_proj4: str = DEFAULT_ROTATED_PROJ4,
    refine_factor: int = 5,
    refine_buffer_m: int = 1200,
    subset_buffer_m: int = 2000,
    return_dataset: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, xr.Dataset]:
    """
    Build normalized grid weights for MEPS over a catchment polygon.

    Returns a DataFrame with columns ['y','x','normalized_grid_weights'].
    """
    # Load dataset if not provided
    if ds is None:
        ds = xr.open_dataset(ds_url)

    # Load & project shape to the rotated grid CRS
    catchment_rot = load_and_project_shape(shapefile_path, rotated_proj4)

    # Subset ds around the shape bbox
    ds_subset = subset_dataset(ds, catchment_rot, buffer_m=subset_buffer_m)

    # Build a refined grid then keep points within polygon
    x_fine, y_fine = refine_grid(
        ds_subset.x.values, ds_subset.y.values,
        factor=refine_factor, buffer_m=refine_buffer_m
    )
    points = [Point(x, y) for x, y in zip(x_fine.ravel(), y_fine.ravel())]
    grid_gdf = gpd.GeoDataFrame(geometry=points, crs=catchment_rot.crs)

    poly_union = catchment_rot.union_all()
    filtered = grid_gdf[grid_gdf.within(poly_union)]

    # Accumulate nearest-cell counts
    ox, oy = ds_subset.x.values, ds_subset.y.values
    weights = np.zeros((len(oy), len(ox)), dtype=np.float64)

    # nearest neighbor per point
    for p in filtered.geometry:
        ix = int(np.argmin(np.abs(ox - p.x)))
        iy = int(np.argmin(np.abs(oy - p.y)))
        weights[iy, ix] += 1.0

    total = weights.sum()
    if total == 0:
        # no points fell inside (edge case) -> leave zeros
        normalized = weights
    else:
        normalized = weights / total

    ds_subset["normalized_grid_weights"] = xr.DataArray(
        data=normalized, dims=["y", "x"], coords={"y": oy, "x": ox}
    )

    df = ds_subset.isel(time=0).to_dataframe().reset_index()[["y", "x", "normalized_grid_weights"]]

    return df

# ---------- optional CLI ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute MEPS normalized grid weights for a catchment.")
    parser.add_argument("--shapefile", required=True)
    parser.add_argument("--ds-url", default=DEFAULT_DS_URL)
    parser.add_argument("--refine-factor", type=int, default=5)
    parser.add_argument("--refine-buffer-m", type=int, default=1200)
    parser.add_argument("--subset-buffer-m", type=int, default=2000)
    args = parser.parse_args()

    df = compute_coords_weights(
        shapefile_path=args.shapefile,
        ds=None,
        ds_url=args.ds_url,
        refine_factor=args.refine_factor,
        refine_buffer_m=args.refine_buffer_m,
        subset_buffer_m=args.subset_buffer_m,
    )
