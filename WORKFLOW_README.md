# Inflow Forecast Workflow

This README summarizes the three command-line flows added to support MEPS caching, onboarding new basins, and running forecasts. All commands assume you are in the repository root (`~/path_to_repo/Inflow-forecast`).

## 1. Cache MEPS Forcings

Prefetch raw MEPS NetCDF files so repeated runs avoid re-downloading.

```bash
python download_meps.py   --start 2025-10-27   --end 2025-10-29   --init-hours 0,6   --leadtimes 0,1,2,3,4,5,6   --cache-dir Data/meps_cache
```

- Adjust `--start/--end` to control the date range (inclusive).
- `--init-hours` (UTC) and `--leadtimes` (hours) should match what your forecast jobs will need.
- Use cron/systemd to automate this every 3 or 24 hours as desired ()see below


## 2. Onboard a New Basin

Run the helper to ingest static info, compute weights, and build the initial time series:

```bash
python run_include_new_catchment.py   --basin ytre_alsaaker   --meps-cache Data/meps_cache
```

This expects the PDF (`pdfs/Nedbørfeltparam-<basin>.pdf`) and shapefile (`shps/<basin>/NedbfeltF_v4.shp`) to exist (get them from Nevina). The script appends static attributes, writes `coords_weights/<basin>_coords_weights.csv`, and updates `Data/time_series/<basin>.nc` using the cached MEPS data.

## 3. Generate a Forecast

With the cache primed and the basin onboarded, run the forecast pipeline:

```bash
python run_forecast.py   --basin ytre_alsaaker   --data-path Data   --model-path inflow_model/model   --scaler-path inflow_model/scaler.pickle   --forecast-dir forecasts   --meps-cache Data/meps_cache
```

The job will update forcings(=features for non-metereoligists) for today, evaluate the MFLSTM, and write a CSV like `forecasts/<date>_ytre_alsaaker.csv` containing observed/predicted discharge in mm/h and cumecs.


## Set up CRON
```
# Run every 3 hours to keep MEPS cache warm
# (edit paths/hours/lead times to match your environment)
0 */3 * * * /usr/bin/env bash -lc 'cd /home/USER/path/to/Inflow-forecast && python download_meps.py --start $(date +\%Y-\%m-\%d --date="-1 day") --end $(date +\%Y-\%m-\%d) --init-hours 0,6,12,18 --leadtimes 0,1,2,3,4,5,6 --cache-dir Data/meps_cache >> logs/meps_prefetch.log 2>&1'
```

Every 3 hours, go into `/home/USER/path/to/Inflow-forecast`, run `download_meps.py` for yesterday through today’s forecasts, and append the output to `logs/meps_prefetch.log`.

> **Tips**
> - Ensure the cache exists (`mkdir -p Data/meps_cache`) before first use.
> - If you change the init hours or lead times used in forecasts, keep the prefetch script in sync.

