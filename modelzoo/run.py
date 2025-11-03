# Import necessary packages
import argparse
import pickle
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import torch
import xarray as xr
from torch.utils.data import DataLoader

from .hy2dl.aux_functions.utils import upload_to_device
from .hy2dl.datasetzoo.camelsno import CAMELS_NO as Datasetclass
from .hy2dl.modelzoo.mflstm import MFLSTM as modelclass
from .add_meps import update_timeseries
from .model_setup import dynamic_input, load_model, static_input, target


def _ensure_forecast_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _adapt_scaler_if_needed(scaler: dict) -> dict:
    try:
        dyn_keys = (
            dynamic_input
            if isinstance(dynamic_input, list)
            else list(
                {
                    k
                    for values in dynamic_input.values()
                    for k in (values if isinstance(values, list) else list(values))
                }
            )
        )

        def _to_dict(vec):
            if isinstance(vec, dict):
                return vec
            if isinstance(vec, torch.Tensor):
                if vec.ndim == 0:
                    return {k: vec for k in dyn_keys}
                if vec.numel() == len(dyn_keys):
                    return {k: vec[i] for i, k in enumerate(dyn_keys)}
            return vec

        if "x_d_mean" in scaler and not isinstance(scaler["x_d_mean"], dict):
            scaler["x_d_mean"] = _to_dict(scaler["x_d_mean"])
        if "x_d_std" in scaler and not isinstance(scaler["x_d_std"], dict):
            scaler["x_d_std"] = _to_dict(scaler["x_d_std"])
    except Exception:
        pass
    return scaler


def run_forecast(
    basin_id: str,
    data_path: Path = Path("Data"),
    model_path: Path = Path("inflow_model/model"),
    scaler_path: Path = Path("inflow_model/scaler.pickle"),
    forecast_dir: Path = Path("forecasts"),
    meps_cache_dir: Path = Path("Data/meps_cache"),
):
    start = time.time()

    basin_id = basin_id.strip()
    basin_lower = basin_id.lower()
    data_path = Path(data_path)
    forecast_dir = Path(forecast_dir)
    meps_cache_dir = Path(meps_cache_dir)

    today = date.today()
    fcast_end = today + timedelta(days=2)
    fcast_period = [f"{today - timedelta(days=366)} 00:00:00", f"{fcast_end} 23:00:00"]

    time_series_path = data_path / "time_series" / f"{basin_lower}.nc"
    weights_path = Path("coords_weights") / f"{basin_id}_coords_weights.csv"
    attr_path = data_path / "attributes" / "attributes.csv"
    attr = pd.read_csv(attr_path)

    _ensure_forecast_dir(forecast_dir)
    save_path = forecast_dir / f"{today}_{basin_lower}.csv"

    # Update historical + forecast forcings
    update_timeseries(
        nora_nc_path=str(time_series_path),
        weights_csv_path=str(weights_path),
        out_nc_path=str(time_series_path),
        start_date=today,
        end_date=today,
        init_hours=[0],
        leadtimes=list(range(0, 6)),
        cache_dir=meps_cache_dir,
    )

    try:
        update_timeseries(
            nora_nc_path=str(time_series_path),
            weights_csv_path=str(weights_path),
            out_nc_path=str(time_series_path),
            start_date=today,
            end_date=today,
            init_hours=[6],
            leadtimes=list(range(0, 67)),
            cache_dir=meps_cache_dir,
        )
    except Exception as e:
        print(f"init_hours=[6] failed: {e} -> trying [3]")
        update_timeseries(
            nora_nc_path=str(time_series_path),
            weights_csv_path=str(weights_path),
            out_nc_path=str(time_series_path),
            start_date=today,
            end_date=today,
            init_hours=[3],
            leadtimes=list(range(6, 67)),
            cache_dir=meps_cache_dir,
        )

    # Fill NaNs conservatively using mean over available history
    ds = xr.load_dataset(time_series_path)
    vars_to_fill = [
        "air_pressure_at_sea_level",
        "air_temperature_2m",
        "precipitation_amount_hourly",
        "relative_humidity_2m",
        "surface_net_longwave_radiation",
        "surface_net_shortwave_radiation",
        "wind_direction",
        "wind_speed",
        "low_type_cloud_area_fraction",
        "medium_type_cloud_area_fraction",
    ]
    for v in vars_to_fill:
        if v in ds:
            ds[v] = ds[v].fillna(ds[v].mean(dim="date", skipna=True))
    ds.to_netcdf(time_series_path, mode="w")

    # Load model + scaler
    model, device, model_configuration, _ = load_model(
        modelclass=modelclass,
        model_path=model_path,
        running_device="gpu",
        strict=False,
    )
    print("Device:", device)

    with open(scaler_path, "rb") as file:
        scaler = pickle.load(file)
    scaler = _adapt_scaler_if_needed(scaler)

    dataset = Datasetclass(
        dynamic_input=dynamic_input,
        forcing=["nora3"],
        target=target,
        sequence_length=model_configuration["seq_length"],
        time_period=fcast_period,
        path_data=str(data_path),
        entity=basin_lower,
        check_NaN=False,
        predict_last_n=model_configuration["predict_last_n_evaluation"],
        static_input=static_input,
        custom_freq_processing=model_configuration["custom_freq_processing"],
        dynamic_embedding=model_configuration["dynamic_embeddings"],
        unique_prediction_blocks=model_configuration["unique_prediction_blocks"],
    )
    dataset.scaler = scaler
    dataset.standardize_data(standardize_output=False)

    model.eval()
    df_ts = pd.DataFrame()
    with torch.no_grad():
        loader = DataLoader(
            dataset=dataset,
            batch_size=model_configuration["batch_size_evaluation"],
            shuffle=False,
            drop_last=False,
            collate_fn=dataset.collate_fn,
        )

        for sample in loader:
            sample = upload_to_device(sample, device)
            pred = model(sample)
            y_sim = pred["y_hat"] * dataset.scaler["y_std"].to(device) + dataset.scaler["y_mean"].to(device)

            df = pd.DataFrame(
                {
                    "y_obs": sample["y_obs"].flatten().cpu().detach(),
                    "y_sim": y_sim[:, -model_configuration["predict_last_n_evaluation"] :, :].flatten().cpu().detach(),
                },
                index=pd.to_datetime(sample["date"].flatten()),
            )
            df_ts = pd.concat([df_ts, df], axis=0)

            del sample, pred, y_sim
            torch.cuda.empty_cache()

    area_row = attr.loc[attr["basin_id"] == basin_lower, "area_total"]
    if area_row.empty:
        raise ValueError(f"No area_total found in attributes for basin '{basin_lower}'.")
    area = area_row.values[0]

    df_out = df_ts.reset_index().rename(columns={"index": "date"})
    df_out["basin_id"] = basin_lower
    df_out["area_km2"] = area
    df_out["streamflow_obs_cumecs"] = df_out["y_obs"] * area / 3.6
    df_out["streamflow_sim_cumecs"] = df_out["y_sim"] * area / 3.6
    df_out.to_csv(save_path, index=False)

    end = time.time()
    print(f"Forecast saved to {save_path} | Process time: {round((end - start) / 60, 1)}min")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run inflow forecast for a basin.")
    parser.add_argument("--basin", default="ytre_alsaaker", help="Basin identifier (lowercase).")
    parser.add_argument("--data-path", default="Data", help="Root data folder (contains time_series/ and attributes/).")
    parser.add_argument("--model-path", default="inflow_model/model", help="Path to trained model weights.")
    parser.add_argument("--scaler-path", default="inflow_model/scaler.pickle", help="Path to scaler pickle.")
    parser.add_argument("--forecast-dir", default="forecasts", help="Directory to write forecast CSV.")
    parser.add_argument("--meps-cache", default="Data/meps_cache", help="Directory to cache raw MEPS NetCDF files.")
    args = parser.parse_args(argv)
    run_forecast(
        basin_id=args.basin,
        data_path=Path(args.data_path),
        model_path=Path(args.model_path),
        scaler_path=Path(args.scaler_path),
        forecast_dir=Path(args.forecast_dir),
        meps_cache_dir=Path(args.meps_cache),
    )


if __name__ == "__main__":
    main()
