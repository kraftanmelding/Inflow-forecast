from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd
import xarray as xr


MEPS_URL_TEMPLATE = (
    "https://thredds.met.no/thredds/dodsC/"
    "meps25epsarchive/{:%Y}/{:%m}/{:%d}/{:%H}/member_00/"
    "meps_sfc_{:02d}_{:%Y%m%dT%H}Z.nc"
)


def _local_path(cache_dir: Path, init_dt: pd.Timestamp, leadtime: int) -> Path:
    return cache_dir / f"{init_dt:%Y/%m/%d/%H}" / f"meps_sfc_{leadtime:02d}_{init_dt:%Y%m%dT%H}Z.nc"


def fetch_meps_dataset(
    init_dt: pd.Timestamp,
    leadtime: int,
    cache_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Tuple[xr.Dataset, str]:
    """
    Retrieve a single MEPS forecast file, optionally caching it locally.

    Parameters
    ----------
    init_dt : pd.Timestamp
        Forecast initialization timestamp (UTC).
    leadtime : int
        Lead time (hours) relative to init_dt.
    cache_dir : Path, optional
        Directory where NetCDF files should be cached. If supplied, the function
        will reuse existing files and write new downloads into this folder.
    verbose : bool
        Whether to print status messages when downloading or reusing cached files.

    Returns
    -------
    (dataset, source) : Tuple[xr.Dataset, str]
        The opened Dataset and a human-readable description of the source
        (cache path or URL).
    """

    url = MEPS_URL_TEMPLATE.format(init_dt, init_dt, init_dt, init_dt, leadtime, init_dt)

    if cache_dir:
        local_path = _local_path(Path(cache_dir), init_dt, leadtime)
        if local_path.exists():
            if verbose:
                print(f"MEPS cache hit: {local_path}")
            return xr.open_dataset(local_path), str(local_path)

    if verbose:
        print(f"Downloading MEPS: {url}")
    ds_remote = xr.open_dataset(url)
    ds_remote.load()

    # Clip to 4.7 <= lon <= 31.1 and 57.98 <= lat <= 71.165 (W & N Norway)
    lon_min, lon_max = 4.7, 31.1
    lat_min, lat_max = 57.98, 71.165
    if {"longitude", "latitude"}.issubset(ds_remote.coords):
        ds_remote = ds_remote.sel(
            latitude=slice(lat_min, lat_max),
            longitude=slice(lon_min, lon_max),
        )
    else:
        lon2d, lat2d = xr.broadcast(ds_remote["x"], ds_remote["y"])
        mask = (
            (lat2d >= lat_min)
            & (lat2d <= lat_max)
            & (lon2d >= lon_min)
            & (lon2d <= lon_max)
        )
        ds_remote = ds_remote.isel(y=mask.any(axis=1), x=mask.any(axis=0))

    if cache_dir:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        ds_remote.to_netcdf(local_path)
        ds_remote.close()
        if verbose:
            print(f"Saved MEPS cache: {local_path}")
        return xr.open_dataset(local_path), str(local_path)

    return ds_remote, url


def prefetch_meps_range(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    init_hours: Iterable[int],
    leadtimes: Iterable[int],
    cache_dir: Path,
    verbose: bool = True,
) -> None:
    """
    Download a block of MEPS forecast files into the cache.

    Parameters
    ----------
    start_date, end_date : pd.Timestamp
        Date range (inclusive) to cover.
    init_hours : iterable of int
        Initialization hours (UTC) to fetch for each day.
    leadtimes : iterable of int
        Lead times (hours) to fetch.
    cache_dir : Path
        Destination directory for cached NetCDF files.
    verbose : bool
        Print progress information.
    """
    cache_dir = Path(cache_dir)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    for date in dates:
        for ih in init_hours:
            init_dt = pd.Timestamp(date).replace(hour=ih)
            for lt in leadtimes:
                try:
                    ds, source = fetch_meps_dataset(init_dt, lt, cache_dir=cache_dir, verbose=verbose)
                    ds.close()
                except Exception as err:
                    if verbose:
                        print(f"Failed to fetch {init_dt:%Y-%m-%d %H} +{lt:02d}h: {err}")
