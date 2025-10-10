# model_setup.py
from pathlib import Path
import torch

# ---- MODEL INPUT ----
dynamic_input = [
    'air_pressure_at_sea_level','air_temperature_2m','precipitation_amount_hourly',
    'relative_humidity_2m','surface_net_longwave_radiation','surface_net_shortwave_radiation',
    'wind_direction','wind_speed','low_type_cloud_area_fraction','medium_type_cloud_area_fraction'
]
 
target = ["QObs(mm/h)"]

static_input = [
    'longitude','latitude','length_km_river','area_total','height_minimum','height_maximum',
    'height_hypso_10','height_hypso_20','height_hypso_30','height_hypso_40','height_hypso_50',
    'height_hypso_60','height_hypso_70','height_hypso_80','height_hypso_90','specific_runoff',
    'perc_lake','perc_forest','perc_mountain','perc_agricul','perc_bog','perc_eff_lake',
    'perc_glacier','perc_urban','is_sk'
]

# ---- DEFAULT CONFIG ----
def build_config(dynamic_input, static_input):
    # model configuration
    cfg = {
        "n_dynamic_channels_lstm": 10,
        "no_of_layers": 1,
        "seq_length": 365 * 24,  # 1 year of hourly data
        "custom_freq_processing": {
            "1D": {"n_steps": 351,"freq_factor": 24,},
            "1h": {"n_steps": (365 - 351) * 24, "freq_factor": 1}},
        "predict_last_n": 24,
        "unique_prediction_blocks": True,
        "dynamic_embeddings": False,
        "hidden_size": 128,
        "batch_size_training": 64,
        "batch_size_evaluation": 256,
        "no_of_epochs": 4,
        "dropout_rate": 0.4, 
        "learning_rate": 0.0005,
        "set_forget_gate": 3,
        "validate_every": 1,
        "validate_n_random_basins": 1,
        "early_stopping_patience": 10,
        "weight_decay": 0
    }

    
    cfg["dynamic_input_size"] = len(dynamic_input) if isinstance(dynamic_input, list) \
        else {k: len(v) for k, v in dynamic_input.items()}
    cfg["input_size_lstm"] = cfg["n_dynamic_channels_lstm"] + len(static_input)
    if cfg.get("custom_freq_processing") and not cfg.get("dynamic_embeddings"):
        cfg["input_size_lstm"] += 1
    cfg["predict_last_n_evaluation"] = 1 if (cfg.get("predict_last_n", 1) > 1
                                             and not cfg.get("unique_prediction_blocks")) \
                                            else cfg.get("predict_last_n", 1)
    return cfg

#GPU /CPU
def select_device(running_device: str = "gpu") -> str:
    if running_device == "gpu" and torch.cuda.is_available():
        # avoid printing during import; caller can print if desired
        return "cuda:0"
    return "cpu"


def load_model(modelclass, model_path: str | Path, running_device: str = "gpu",
               strict: bool = True):
    """
    Returns (model, device, config).
    """
    device = select_device(running_device)
    cfg = build_config(dynamic_input, static_input)

    model = modelclass(model_configuration=cfg).to(device)

    ckpt_path = Path(model_path)
    state = torch.load(ckpt_path, map_location=device)
    # handle wrappers (Lightning/DataParallel) gracefully
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
    else:
        state = {k.replace("module.", ""): v for k, v in state.items()}

    missing_unexp = model.load_state_dict(state, strict=strict)
    return model, device, cfg, missing_unexp

# What to export if someone uses `from model_setup import *`
__all__ = ["dynamic_input", "static_input", "target", "build_config", "load_model", "select_device"]
