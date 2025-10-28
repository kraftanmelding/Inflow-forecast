# Import necessary packages
import pickle
import random
import sys
import time
import tempfile
import os
from datetime import date,timedelta
import time
import xarray as xr
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import classes and functions from local package (explicit package path)
from modelzoo.hy2dl.aux_functions.utils import upload_to_device
from modelzoo.hy2dl.datasetzoo.camelsno import CAMELS_NO as Datasetclass
from modelzoo.hy2dl.modelzoo.mflstm import MFLSTM as modelclass

from modelzoo.add_meps import update_timeseries
from modelzoo.model_setup import load_model, dynamic_input, static_input, target


#Model paths #s
model_path = "inflow_model/model"
scaler_path = "inflow_model/scaler.pickle"

#Data paths
data_path = "Data"
attr = pd.read_csv("Data/attributes/attributes.csv")

#Plant name
basin_id = 'ytre_alsaaker'#Holmen'

today = date.today()
fcast_end = today + timedelta(days=2)

save_path = f"forecasts/{today}_{basin_id.lower()}.csv"
os.makedirs("forecasts", exist_ok=True)

#include warmup period of 1 year
fcast_period = [f"{today - timedelta(days=366)} 00:00:00", f"{fcast_end} 23:00:00"]


start = time.time()

# add forecast data for today and tomorrow
update_timeseries(
    nora_nc_path=f"{data_path}/time_series/{basin_id.lower()}.nc",
    weights_csv_path=f"coords_weights/{basin_id}_coords_weights.csv",
    out_nc_path=f"{data_path}/time_series/{basin_id.lower()}.nc",  
    start_date=today,
    end_date=today,
    init_hours=[0],
    leadtimes=list(range(0,6)),
)

try:
    update_timeseries(
        nora_nc_path=f"{data_path}/time_series/{basin_id.lower()}.nc",
        weights_csv_path=f"coords_weights/{basin_id}_coords_weights.csv",
        out_nc_path=f"{data_path}/time_series/{basin_id.lower()}.nc",  
        start_date=today,
        end_date=today,
        init_hours=[6],
        leadtimes=list(range(0, 67)),
    )

#if 6z not available, try 3z
except Exception as e:
    print(f"init_hours=[6] failed: {e} -> trying [3]")
    update_timeseries(
        nora_nc_path=f"{data_path}/time_series/{basin_id.lower()}.nc",
        weights_csv_path=f"coords_weights/{basin_id}_coords_weights.csv",
        out_nc_path=f"{data_path}/time_series/{basin_id.lower()}.nc",  
        start_date=today,
        end_date=today,
        init_hours=[3],
        leadtimes=list(range(6, 67)),
    )


# Fill NaNs (to be improved in the future)
ds = xr.load_dataset(f"{data_path}/time_series/{basin_id.lower()}.nc")

vars_to_fill = [
    "air_pressure_at_sea_level","air_temperature_2m","precipitation_amount_hourly",
    "relative_humidity_2m","surface_net_longwave_radiation","surface_net_shortwave_radiation",
    "wind_direction","wind_speed","low_type_cloud_area_fraction","medium_type_cloud_area_fraction",
]

for v in vars_to_fill:
    if v in ds:
        ds[v] = ds[v].fillna(ds[v].mean(dim="date", skipna=True))


# overwrite safely now that the source is closed
ds.to_netcdf(f"{data_path}/time_series/{basin_id.lower()}.nc", mode="w")


#set up model
model, device, model_configuration, (missing, unexpected) = load_model(
    modelclass=modelclass,
    model_path=model_path,
    running_device="gpu",   # or "cpu"
    strict=False            # set True if config matches exactly
)

print("Device:", device)


with open(scaler_path, "rb") as file:
    scaler = pickle.load(file)

# Adapt scaler format if coming from older training (vector -> per-variable dict)
try:
    import torch
    def _to_dict(vec, keys):
        if isinstance(vec, dict):
            return vec
        if isinstance(vec, torch.Tensor):
            if vec.ndim == 0:
                return {k: vec for k in keys}
            if vec.numel() == len(keys):
                return {k: vec[i] for i, k in enumerate(keys)}
        return vec
    dyn_keys = dynamic_input if isinstance(dynamic_input, list) else list({k for v in dynamic_input.values() for k in (v if isinstance(v, list) else list(v))})
    if "x_d_mean" in scaler and not isinstance(scaler["x_d_mean"], dict):
        scaler["x_d_mean"] = _to_dict(scaler["x_d_mean"], dyn_keys)
    if "x_d_std" in scaler and not isinstance(scaler["x_d_std"], dict):
        scaler["x_d_std"] = _to_dict(scaler["x_d_std"], dyn_keys)
except Exception as _e:
    pass


dataset = Datasetclass(
    dynamic_input=dynamic_input,
    forcing=['nora3'],
    target=target,
    sequence_length=model_configuration["seq_length"],
    time_period=fcast_period,
    path_data=data_path,
    entity=basin_id.lower(),
    check_NaN=False,
    predict_last_n=model_configuration["predict_last_n_evaluation"],
    static_input=static_input,
    custom_freq_processing=model_configuration["custom_freq_processing"],
    dynamic_embedding=model_configuration["dynamic_embeddings"],
    unique_prediction_blocks=model_configuration["unique_prediction_blocks"]
)

dataset.scaler = scaler
dataset.standardize_data(standardize_output=False)
testing_dataset = dataset


model.eval()

with torch.no_grad():
    loader = DataLoader(
        dataset=dataset,
        batch_size=model_configuration["batch_size_evaluation"],
        shuffle=False,
        drop_last=False,
        collate_fn=testing_dataset.collate_fn,
    )

    df_ts = pd.DataFrame()
    for sample in loader:
        sample = upload_to_device(sample, device)  # upload tensors to device
        pred = model(sample)
        # backtransformed information
        y_sim = pred["y_hat"] * dataset.scaler["y_std"].to(device) + dataset.scaler["y_mean"].to(device)

        # join results in a dataframe and store them in a dictionary (is easier to plot later)
        df = pd.DataFrame(
            {
                "y_obs": sample["y_obs"].flatten().cpu().detach(),
                "y_sim": y_sim[:, -model_configuration["predict_last_n_evaluation"] :, :].flatten().cpu().detach(),
            },
            index=pd.to_datetime(sample["date"].flatten()),
        )

        df_ts = pd.concat([df_ts, df], axis=0)

        # remove from cuda
        del sample, pred, y_sim
        torch.cuda.empty_cache()

    test_results = df_ts


area_row = attr.loc[attr['basin_id'] == basin_id.lower(), 'area_total']
area = area_row.values[0]

# Prepare DataFrame
df_out = test_results.copy()
df_out = df_out.reset_index().rename(columns={'index': 'date'})
df_out['basin_id'] = basin_id.lower()
df_out['area_km2'] = area

# Convert to cumecs
df_out['streamflow_obs_cumecs'] = df_out['y_obs'] * area / 3.6
df_out['streamflow_sim_cumecs'] = df_out['y_sim'] * area / 3.6

df_out.to_csv(save_path, index=False)
end = time.time()

print(f'Forecast saved to {save_path} | Process time: {round((end-start)/60,1)}min')
