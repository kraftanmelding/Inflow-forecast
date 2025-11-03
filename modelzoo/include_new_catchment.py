# import libraries
import argparse
import time
from pathlib import Path

import pandas as pd
from pyproj import Transformer

from .pdf_csv import make_df
from .weights_coords import compute_coords_weights
from .add_meps import update_timeseries


KEEP_COLUMNS = [
    "basin_id",
    "latitude",
    "longitude",
    "length_km_river",
    "perc_agricul",
    "perc_bog",
    "perc_eff_lake",
    "perc_lake",
    "perc_glacier",
    "perc_forest",
    "perc_mountain",
    "perc_urban",
    "area_total",
    "height_minimum",
    "height_maximum",
    "height_hypso_10",
    "height_hypso_20",
    "height_hypso_30",
    "height_hypso_40",
    "height_hypso_50",
    "height_hypso_60",
    "height_hypso_70",
    "height_hypso_80",
    "height_hypso_90",
    "specific_runoff",
    "annual_runoff",
    "is_sk",
]


def _extract_static_attributes(pdf_path: Path, basin_id: str) -> pd.DataFrame:
    df = make_df([str(pdf_path)], pattern_prefix="Nedbørfeltparam-")
    transformer = Transformer.from_crs("EPSG:32633", "EPSG:4326", always_xy=True)

    utm_east = df.iloc[0]["utm_east_z33"]
    utm_north = df.iloc[0]["utm_north_z33"]

    lon, lat = transformer.transform(utm_east, utm_north)
    df["latitude"] = lat
    df["longitude"] = lon
    df["basin_id"] = basin_id.lower()
    df["is_sk"] = 1

    return df[KEEP_COLUMNS]


def process_catchment(basin_id: str, meps_cache_dir: Path = Path("Data/meps_cache")):
    start = time.time()

    basin_id = basin_id.strip()
    basin_dir = Path("pdfs") / f"Nedbørfeltparam-{basin_id}.pdf"
    shapefile = Path("shps") / basin_id / "NedbfeltF_v4.shp"

    # Update attributes table
    attr_path = Path("Data/attributes/attributes.csv")
    attr = pd.read_csv(attr_path)
    attr = attr[attr["basin_id"] != basin_id.lower()]

    df = _extract_static_attributes(basin_dir, basin_id)
    updated_attr = pd.concat([df, attr], ignore_index=True)
    updated_attr = updated_attr.drop("Unnamed: 0", axis=1, errors="ignore")
    updated_attr.to_csv(attr_path, index=False)
    print("Attributes updated")

    # Update coordinates/weights
    coords_path = Path("coords_weights") / f"{basin_id}_coords_weights.csv"
    df_coords = compute_coords_weights(shapefile_path=str(shapefile))
    df_coords.to_csv(coords_path, index=False)
    print("Coordinates and weights saved")

    # Update NORA/MEPS time series
    ts_path = Path("Data/time_series") / f"{basin_id.lower()}.nc"
    update_timeseries(
        nora_nc_path=str(ts_path),
        weights_csv_path=str(coords_path),
        out_nc_path=str(ts_path),
        start_date="2024-10-01",
        cache_dir=meps_cache_dir,
    )

    end = time.time()
    print(f"Process time: {round((end - start) / 3600, 1)}h")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Add a new catchment to the inflow pipeline.")
    parser.add_argument("--basin", default="ytre_alsaaker", help="Catchment/basin identifier (matches filenames).")
    parser.add_argument("--meps-cache", default="Data/meps_cache", help="Directory to cache raw MEPS NetCDF files.")
    args = parser.parse_args(argv)
    process_catchment(args.basin, Path(args.meps_cache))


if __name__ == "__main__":
    main()
