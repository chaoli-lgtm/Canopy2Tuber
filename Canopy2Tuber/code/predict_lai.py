"""Command-line LAI inference using the deployed MST-Net checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from model import DualStreamMSTNet
from utils import inverse_transform_lai, load_config, normalize_spectra, regression_metrics, validate_spectral_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canopy2Tuber: predict potato LAI from five-band UAV spectra.")
    parser.add_argument("--input", required=True, help="Input CSV with Sample_ID, Blue, Green, Red, Red_edge, NIR.")
    parser.add_argument("--weights", required=True, help="MST-Net .pth checkpoint.")
    parser.add_argument("--config", required=True, help="JSON-compatible YAML configuration file.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Inference batch size (default: 1024).")
    return parser.parse_args()


def load_model(weights_path: Path, config: dict) -> DualStreamMSTNet:
    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights file was not found: {weights_path}")
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError("Checkpoint must contain a 'model_state' entry.")

    input_dim = int(checkpoint.get("input_dim", -1))
    expected_dim = int(config["input_dimension"])
    if input_dim != expected_dim:
        raise ValueError(f"Checkpoint input_dim={input_dim}, but config expects {expected_dim}.")
    bands = checkpoint.get("bands")
    if bands is not None and list(bands) != list(config["input_bands"]):
        raise ValueError(f"Checkpoint bands={bands}, but config bands={config['input_bands']}.")

    model = DualStreamMSTNet(input_dim=input_dim, d_model=int(config["model_hyperparameters"]["d_model"]))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    input_path, output_path = Path(args.input), Path(args.output)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV was not found: {input_path}")

    config = load_config(args.config)
    frame = validate_spectral_input(pd.read_csv(input_path))
    normalized = normalize_spectra(frame, config)
    model = load_model(Path(args.weights), config)

    batches = []
    with torch.no_grad():
        for start in range(0, len(normalized), args.batch_size):
            batch = torch.from_numpy(normalized[start : start + args.batch_size])
            batches.append(model(batch).cpu().numpy())
    lai = inverse_transform_lai(np.vstack(batches), config)
    if not np.isfinite(lai).all():
        raise RuntimeError("Inference produced NaN or infinite LAI predictions.")

    output = pd.DataFrame({"Sample_ID": frame["Sample_ID"], "Predicted_LAI": lai})
    if "Measured_LAI" in frame.columns:
        observed = pd.to_numeric(frame["Measured_LAI"], errors="coerce")
        valid = observed.notna()
        if valid.sum() >= 2:
            metrics = regression_metrics(observed[valid], lai[valid.to_numpy()])
            for key, value in metrics.items():
                output[key] = value
            print("Measured_LAI evaluation: " + ", ".join(f"{key}={value:.4f}" for key, value in metrics.items()))
        else:
            print("Measured_LAI is present but fewer than two numeric observations are available; metrics were skipped.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Saved {len(output)} predictions to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
