# meps_timeseries.py
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from .meps_cache import fetch_meps_dataset


# --- Public API ----------------------------------------------------------------

def update_timeseries(
    nora_nc_path: str,
    weights_csv_path: str,
    out_nc_path: Optional[str] = None,
    start_date: str = "2024-10-01",
    end_date: Optional[str] = None,
    init_hours: Sequence[int] = (0, 3, 6, 9, 12, 15, 18, 21),
    leadtimes: Sequence[int] = (0, 1, 2),
    cache_dir: str,
    verbose: bool = True,
) -> xr.Dataset:
    """
    Fetch MEPS surface fields, compute a spatially weighted mean using your coords+weights,
    convert precipitation from accumulated to hourly, then append to an existing NORA3
    time series and save.

    Parameters
    ----------
    nora_nc_path : str
        Path to existing NORA3 time series NetCDF (with dim/coord named 'date').
    weights_csv_path : str
        CSV with columns ['y','x','normalized_grid_weights'] for your site.
    out_nc_path : str, optional
        Where to write the updated NetCDF. Defaults to overwrite `nora_nc_path`.
    start_date : str
        ISO date (YYYY-MM-DD) for the first day to fetch MEPS.
    end_date : str, optional
        ISO date (YYYY-MM-DD). Default: yesterday (local).
    init_hours : sequence of int
        Forecast initialization hours (UTC) to pull.
    leadtimes : sequence of int
        Lead times (hours) relative to each initialization.
    cache_dir : str
        Directory where raw MEPS NetCDF files should be cached. This must be provided
        so downloads are reused across basins and subsequent runs.
    verbose : bool
        Print progress.

    Returns
    -------
    xr.Dataset
        The merged NORA+MEPS dataset on the 'date' dimension.
    """
    if not cache_dir:
        raise ValueError("cache_dir must be provided.")

    if out_nc_path is None:
        out_nc_path = nora_nc_path

    # Normalize cache directory
    cache_path = Path(cache_dir)

    # Load existing NORA3 series if present (assumed to use 'date' as the time coordinate)
    nora_ds = None
    try:
        nora_ds = xr.open_dataset(nora_nc_path)
        nora_ds = nora_ds.load()   # read data into memory
        nora_ds.close()            # release file handle / lock
    except FileNotFoundError:
        if verbose:
            print(f"Base file not found at {nora_nc_path}; starting from MEPS only.")

    # Load and prep weights grid (as DataArray on y/x)
    w_grid = _load_weights(weights_csv_path)

    # Figure out date range
    start_dt = pd.to_datetime(start_date).normalize()
    if end_date is None:
        end_dt = (pd.Timestamp.now().normalize() - pd.Timedelta(days=1))
    else:
        end_dt = pd.to_datetime(end_date).normalize()

    meps_ds = fetch_meps_weighted_timeseries(
        w_grid=w_grid,
        start_date=start_dt,
        end_date=end_dt,
        init_hours=init_hours,
        leadtimes=leadtimes,
        cache_dir=cache_path,
        verbose=verbose,
    )

    # Harmonize variable names to NORA naming (if needed)
    rename_map = {
        "surface_downwelling_longwave_flux_in_air": "surface_net_longwave_radiation",
        "surface_downwelling_shortwave_flux_in_air": "surface_net_shortwave_radiation",
    }
    meps_ds = meps_ds.rename({k: v for k, v in rename_map.items() if k in meps_ds})

    for c in ("forecast_init", "leadtime"):
        if c in meps_ds.coords:
            meps_ds = meps_ds.reset_coords(c, drop=True)


    # Append to NORA series (or start from MEPS if no base file)
    if nora_ds is None:
        out = meps_ds
    else:
        out = xr.concat([nora_ds, meps_ds], dim="date", combine_attrs="override")
    out = out.sortby("date")
    # Drop duplicate timestamps (keep first)
    out = out.sel(date=~out.indexes["date"].duplicated())
    hourly_index = pd.date_range(out.indexes["date"][0], out.indexes["date"][-1], freq="h")
    out = out.reindex(date=hourly_index)
    
    # --- WRITE VIA TEMP FILE, THEN REPLACE ---
    # ensure directory exists
    _ensure_dir(os.path.dirname(out_nc_path))
    tmp_path = out_nc_path + ".tmp"
    out.to_netcdf(tmp_path, mode="w")   # ensure write mode
    os.replace(tmp_path, out_nc_path)   # atomic move

    if verbose:
        print(f"Saved merged dataset to: {out_nc_path}")

    return out


def fetch_meps_weighted_timeseries(
    w_grid: xr.DataArray,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    init_hours: Iterable[int] = (0, 3, 6, 9, 12, 15, 18, 21),
    leadtimes: Iterable[int] = (0, 1, 2),
    cache_dir: Path,
    verbose: bool = True,
) -> xr.Dataset:
    meps_vars = [
        "air_pressure_at_sea_level",
        "air_temperature_2m",
        "precipitation_amount_acc",
        "relative_humidity_2m",
        "surface_downwelling_longwave_flux_in_air",
        "surface_downwelling_shortwave_flux_in_air",
        "wind_direction",
        "wind_speed",
        "low_type_cloud_area_fraction",
        "medium_type_cloud_area_fraction",
    ]

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    rows = []

    for date in dates:
        for ih in init_hours:
            init_dt = pd.Timestamp(datetime(date.year, date.month, date.day, ih))
            for lt in leadtimes:
                valid_dt = init_dt + pd.Timedelta(hours=lt)
                try:
                    ds, source = fetch_meps_dataset(
                        init_dt=init_dt,
                        leadtime=lt,
                        cache_dir=cache_dir,
                        verbose=verbose,
                    )
                    if verbose:
                        print(f"Processing {valid_dt:%Y-%m-%d %H:%M}  ({source})")
                except Exception as e:
                    if verbose:
                        print(f"  -> skip (fetch failed): {e}")
                    continue

                vars_here = [v for v in meps_vars if v in ds.variables]
                if not vars_here:
                    ds.close()
                    continue

                weights = (
                    w_grid.interp(y=ds["y"], x=ds["x"], method="nearest")
                    .fillna(0.0)
                )
                w = weights / weights.sum(dim=("y", "x"))

                subset = ds[vars_here]
                wmean = xr.Dataset()
                for name, da in subset.data_vars.items():
                    if {"y", "x"}.issubset(da.dims) and np.issubdtype(da.dtype, np.number):
                        wmean[name] = (da * w).sum(dim=("y", "x"))
                    else:
                        wmean[name] = da.squeeze(drop=True)  # pass-through

                wmean = wmean.squeeze(drop=True)

                # ---------- FIX START: attach coords on the 'date' dim ----------
                # make a 1-step 'date' dimension
                wmean = wmean.expand_dims(date=[np.datetime64(valid_dt)])
                # attach forecast_init and leadtime as coords *on that dim*
                wmean = wmean.assign_coords(
                    forecast_init=("date", [np.datetime64(init_dt)]),
                    leadtime=("date", [lt]),
                )
                # ---------- FIX END ----------------------------------------------

                rows.append(wmean)
                ds.close()

    if not rows:
        raise RuntimeError("No MEPS rows were collected. Check dates/hours/URLs.")

    meps = xr.concat(rows, dim="date", combine_attrs="override")

    # Precipitation: ACC -> hourly *within each forecast init* (use groupwise shift)
    if "precipitation_amount_acc" in meps:
        acc = meps["precipitation_amount_acc"]
        # sort to make sure shift is in time order per init
        meps = meps.sortby(["forecast_init", "date"])
        hourly = acc.groupby("forecast_init").map(lambda g: g - g.shift(date=1))
        hourly = hourly.fillna(acc).clip(min=0)  # first step per init; clip tiny negatives
        meps = meps.drop_vars("precipitation_amount_acc").assign(
            precipitation_amount_hourly=hourly
        )

    # if you prefer not to keep these helper coords, drop them here:
    # meps = meps.drop_vars(["forecast_init", "leadtime"])

    return meps


# --- Helpers -------------------------------------------------------------------

def _load_weights(weights_csv_path: str) -> xr.DataArray:
    """
    Loads CSV with columns ['y','x','normalized_grid_weights'] and returns
    an xr.DataArray on (y,x) with those weights.
    """
    df = pd.read_csv(weights_csv_path)
    df = df.groupby(["y", "x"], as_index=False)["normalized_grid_weights"].mean()
    grid = df.pivot(index="y", columns="x", values="normalized_grid_weights")
    da = xr.DataArray(
        data=grid.values,
        dims=("y", "x"),
        coords={"y": grid.index.values, "x": grid.columns.values},
        name="weights",
    )
    return da


def _ensure_dir(d: str) -> None:
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


# --- CLI (optional) ------------------------------------------------------------

if __name__ == "__main__":
    # Minimal CLI for ad-hoc runs
    import argparse

    parser = argparse.ArgumentParser(description="Append MEPS weighted time series to NORA3 file.")
    parser.add_argument("--nora", required=True, help="Path to existing NORA3 NetCDF (with 'date' coord).")
    parser.add_argument("--weights", required=True, help="Path to CSV with coords/weights.")
    parser.add_argument("--out", default=None, help="Output NetCDF path (default: overwrite --nora).")
    parser.add_argument("--start", default="2024-11-01", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD). Defaults to yesterday.")
    parser.add_argument("--init-hours", default="0,3, 6,9,12,15,18,21", help="Comma-separated init hours.")
    parser.add_argument("--leadtimes", default="0,1,2", help="Comma-separated lead times (hours).")
    parser.add_argument("--cache-dir", required=True, help="Directory to cache raw MEPS NetCDF files.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress prints.")

    args = parser.parse_args()
    init_hours = tuple(int(x) for x in args.init_hours.split(",") if x.strip())
    leadtimes = tuple(int(x) for x in args.leadtimes.split(",") if x.strip())

    update_timeseries(
        nora_nc_path=args.nora,
        weights_csv_path=args.weights,
        out_nc_path=args.out,
        start_date=args.start,
        end_date=args.end,
        init_hours=init_hours,
        leadtimes=leadtimes,
        cache_dir=args.cache_dir,
        verbose=not args.quiet,
    )
