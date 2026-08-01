"""Replay a representative source-to-biomass calculation from the source project.

The demo uses one anonymized treatment time series. RUE parameters and organ
allocation fractions are copied from source model outputs; this script does not
fit any new parameters.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "Thermal_time",
    "N_Level",
    "Density",
    "LAI",
    "Incident_PAR_proxy",
    "RUE_Base",
    "Alpha",
    "Beta",
    "Gamma",
    "Soil_N_Used",
    "HI_Pred",
    "Frac_Root_Pred",
    "LVR_Pred",
]
FIPAR_K = 0.5  # Source: 4fipar计算.py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the representative dynamic potato biomass model demo.")
    parser.add_argument("--input", required=True, help="CSV generated from the representative source-model time series.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    return parser.parse_args()


def validate_input(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError("Input CSV is missing required columns: " + ", ".join(missing))
    cleaned = frame.copy()
    for column in REQUIRED_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    invalid = cleaned[REQUIRED_COLUMNS].isna().any(axis=1)
    if invalid.any():
        raise ValueError("Input contains missing or non-numeric model values.")
    if (cleaned[["LAI", "Incident_PAR_proxy"]] < 0).any().any():
        raise ValueError("LAI and Incident_PAR_proxy must be non-negative.")
    for column in ["HI_Pred", "Frac_Root_Pred", "LVR_Pred"]:
        if ((cleaned[column] < 0) | (cleaned[column] > 1)).any():
            raise ValueError(f"{column} must lie between 0 and 1.")
    return cleaned.sort_values("Thermal_time").reset_index(drop=True)


def run_model(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["fIPAR"] = 1.0 - np.exp(-FIPAR_K * result["LAI"])
    result["APAR"] = result["fIPAR"] * result["Incident_PAR_proxy"]

    n_available = result["N_Level"] + result["Soil_N_Used"]
    result["RUE"] = (
        (result["RUE_Base"] + result["Alpha"] * result["Density"])
        * (1.0 - np.exp(-result["Beta"] * n_available))
        - result["Gamma"] * result["Thermal_time"]
    ).clip(lower=0.0)
    result["Total_Biomass"] = result["APAR"] * result["RUE"]

    result["Tuber_Biomass"] = result["Total_Biomass"] * result["HI_Pred"]
    vegetative = result["Total_Biomass"] - result["Tuber_Biomass"]
    result["Root_Biomass"] = vegetative * result["Frac_Root_Pred"]
    shoot = vegetative - result["Root_Biomass"]
    result["Leaf_Biomass"] = shoot * result["LVR_Pred"]
    result["Stem_Biomass"] = shoot - result["Leaf_Biomass"]
    return result[
        [
            "Thermal_time", "LAI", "fIPAR", "APAR", "RUE", "Total_Biomass",
            "Leaf_Biomass", "Stem_Biomass", "Root_Biomass", "Tuber_Biomass",
        ]
    ]


def main() -> None:
    args = parse_args()
    input_path, output_path = Path(args.input), Path(args.output)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV was not found: {input_path}")
    output = run_model(validate_input(pd.read_csv(input_path)))
    values = output.select_dtypes(include=[np.number]).to_numpy()
    if not np.isfinite(values).all():
        raise RuntimeError("Biomass calculation produced NaN or infinite values.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Saved {len(output)} biomass records to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
