"""Shared validation, configuration, normalization, and metric helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Union

import numpy as np
import pandas as pd


REQUIRED_INPUT_COLUMNS = ["Sample_ID", "Blue", "Green", "Red", "Red_edge", "NIR"]


def load_config(config_path: Union[str, Path]) -> Dict[str, object]:
    """Load JSON-compatible YAML using only the Python standard library.

    The supplied ``.yaml`` file is valid JSON, which is a YAML subset. This
    avoids introducing an unneeded YAML parsing dependency into the demo.
    """

    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file was not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Configuration is not valid JSON-compatible YAML: {path}") from exc


def validate_spectral_input(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            "Input CSV is missing required columns: "
            + ", ".join(missing)
            + ". Required order is Blue, Green, Red, Red_edge, NIR."
        )
    cleaned = frame.copy()
    for column in REQUIRED_INPUT_COLUMNS[1:]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    invalid = cleaned[REQUIRED_INPUT_COLUMNS[1:]].isna().any(axis=1)
    if invalid.any():
        rows = ", ".join(str(index + 2) for index in cleaned.index[invalid][:10])
        raise ValueError(f"Non-numeric or missing spectral values found in CSV row(s): {rows}")
    values = cleaned[REQUIRED_INPUT_COLUMNS[1:]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Spectral input contains NaN or infinite values.")
    return cleaned


def normalize_spectra(frame: pd.DataFrame, config: Dict[str, object]) -> np.ndarray:
    params = config["normalization_parameters"]["spectral"]
    mean = np.asarray(params["mean"], dtype=np.float32)
    scale = np.asarray(params["scale"], dtype=np.float32)
    values = frame[REQUIRED_INPUT_COLUMNS[1:]].to_numpy(dtype=np.float32)
    if values.shape[1] != len(mean) or len(mean) != len(scale):
        raise ValueError("Normalization parameters do not match the five required spectral bands.")
    if np.any(scale == 0):
        raise ValueError("Invalid zero scale value in normalization parameters.")
    return (values - mean) / scale


def inverse_transform_lai(prediction: np.ndarray, config: Dict[str, object]) -> np.ndarray:
    params = config["normalization_parameters"]["target"]
    mean = float(params["mean"][0])
    scale = float(params["scale"][0])
    return prediction.reshape(-1) * scale + mean


def regression_metrics(observed: Iterable[float], predicted: Iterable[float]) -> Dict[str, float]:
    observed_array = np.asarray(list(observed), dtype=float)
    predicted_array = np.asarray(list(predicted), dtype=float)
    if len(observed_array) < 2:
        return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan}
    residual = predicted_array - observed_array
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    denominator = float(np.sum((observed_array - observed_array.mean()) ** 2))
    r2 = float(1.0 - np.sum(residual**2) / denominator) if denominator > 0 else np.nan
    return {"R2": r2, "RMSE": rmse, "MAE": mae}
