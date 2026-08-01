# Canopy2Tuber reproducibility package report

## Scope

Canopy2Tuber was assembled as a minimal reproducibility release for an
integrated UAV phenotyping, PROSAIL-informed deep learning, and dynamic crop
modeling framework for potato LAI retrieval and underground tuber yield
prediction. It supports model loading, representative LAI inference, and a
compact biomass-model example, while excluding complete raw field data, UAV
imagery, training code, full experimental parameters, training logs, and
intermediate checkpoints.

## Source-to-package mapping

| Shared file | Source item inspected | Action and verification |
|---|---|---|
| `weights/MSTNet_LAI_best.pth` | `Final_Deployed_Model.pth` | Copied as the only deployment checkpoint. The source deployment script directly references this file. |
| `code/model.py` | `stage2_transformer应用.py` | Extracted the deployed `ResNetBlock1D` and `DualStreamMSTNet` architecture only. |
| `code/predict_lai.py`, `code/utils.py` | Deployment script plus checkpoint metadata | Minimal CSV inference wrapper. It uses the deployed five-band order and checkpoint-derived standardization values. |
| `config/MSTNet_config.yaml` | `Final_Deployed_Model.pth` | Recorded checkpoint keys, band list, input dimension, state-compatible architecture, and `StandardScaler` means/scales. |
| `example_data/example_UAV_spectra.csv` | `real1202.xlsx`, Sheet6 | Fixed-seed (3407) sample of 40 records: eight samples from each of five measured-LAI bins after limiting absolute deployed-model error to 0.85 LAI. The retained LAI range is 1.03--6.40. IDs were replaced with `UAV_001`--`UAV_040`; all non-spectral fields except measured LAI were omitted. This filtered code-testing set is not a random validation sample or a manuscript performance claim. |
| `example_data/example_PROSAIL_spectra.csv` | `prosail1202.xlsx`, Sheet1 | Fixed-seed (3407) random sample of 200 of 10,000 simulations. Only five bands, anonymous IDs, and simulated LAI were retained. |
| `example_data/example_growth_inputs.csv` | Representative records from `2023b4treatment.xlsx`, `2023b4_Final_Date_Model.xlsx`, and `2023b4_Model_Parameters.xlsx` | Four anonymized time points. No source filenames, plot IDs, locations, or personal information are retained. |
| `code/biomass_model_demo.py` | `4fipar计算.py`, `RUE计算.py`, and source model-output tables | Replays the verified fIPAR formula, dynamic RUE equation, and source-provided HI/root/leaf allocation fractions; no parameters are re-estimated. |

## Final weight selection

`Final_Deployed_Model.pth` was selected because it is the only explicitly named
deployment checkpoint at the project root and is loaded by the inspected source
deployment script. Its checkpoint contains `model_state`, `scalers`, `bands`,
and `input_dim`. The package renames this copy to `MSTNet_LAI_best.pth`.

`TODO_AUTHOR_CONFIRM`: Please confirm that `Final_Deployed_Model.pth` is the
weight referred to as the final/best LAI model in the manuscript. No explicit
validation-ranking manifest or model-selection log was found next to the
checkpoint.

## Confirmed model configuration

- Checkpoint bands: `[450, 560, 650, 730, 840]`.
- CSV order: `Blue, Green, Red, Red_edge, NIR`.
- Input dimension: `5`; output dimension: `1` LAI value.
- Architecture: 128-dimensional Transformer embedding, 3 Transformer layers,
  4 heads, parallel 1D residual CNN (64 channels), and fused 64-unit regression
  head.
- Standardization: checkpoint `StandardScaler` values are recorded verbatim in
  `config/MSTNet_config.yaml`.
- Weight loading: `model.load_state_dict(..., strict=True)`.

## Biomass demonstration details

The source fIPAR relation is `fIPAR = 1 - exp(-0.5 × LAI)`. The RUE replay is
the source-model equation:

```text
RUE = max(0, (RUE_Base + Alpha × Density)
              × (1 - exp(-Beta × (N_Level + Soil_N_Used)))
              - Gamma × Thermal_time)
```

The demonstration then calculates total biomass as `APAR × RUE` and divides
that value using source-model `HI_Pred`, `Frac_Root_Pred`, and `LVR_Pred`.
`Incident_PAR_proxy` is reconstructed algebraically from source APAR and the
source fIPAR relation because the full daily incident-PAR time series is not
released in this minimal package. It is not an independently observed PAR
series.

## Test commands and results

Commands were run from the package root with Python 3.8.20 and PyTorch
2.4.1+cpu:

```bash
python code/predict_lai.py --input example_data/example_UAV_spectra.csv --weights weights/MSTNet_LAI_best.pth --config config/MSTNet_config.yaml --output example_outputs/predicted_LAI.csv
python code/biomass_model_demo.py --input example_data/example_growth_inputs.csv --output example_outputs/predicted_biomass.csv
```

| Test | Result |
|---|---|
| Checkpoint loading | Passed with `strict=True`. |
| Five-band input validation | Passed for 40 field example rows. |
| LAI output | Passed; 40 predictions; no NaN or infinite values. |
| Example LAI evaluation | R2 = 0.9230; RMSE = 0.4034; MAE = 0.3149. The example file is a fixed-seed, error-screened code-testing sample (absolute error <= 0.85 LAI), not a random validation sample or a manuscript performance claim. |
| Biomass output | Passed; 4 records with fIPAR, APAR, RUE, total biomass, and four organ compartments; no NaN or infinite values. |

## Files intentionally excluded

- Complete UAV imagery, plot coordinates, GPS information, and other spatial
  identifiers.
- Complete field and simulated datasets.
- Training/tuning scripts, logs, temporary files, and non-final checkpoints.
- Full weather/PAR records and all experiment-specific parameter tables.

## Author confirmation items

1. `TODO_AUTHOR_CONFIRM`: confirm the formal manuscript designation of
   `Final_Deployed_Model.pth` as the final/best LAI checkpoint.
2. `TODO_AUTHOR_CONFIRM`: confirm the training random seed. It was not saved
   in the deployed checkpoint or the inspected deployment script.
3. `TODO_AUTHOR_CONFIRM`: decide and add a formal public license before any
   repository release; no source-project license was found.
4. The compact biomass example reproduces the documented model-output pathway,
   but it cannot independently regenerate APAR from unreleased daily incident
   PAR. The proxy-based demonstration is disclosed in the README and above.

## Privacy check

Plaintext package files were scanned for absolute local paths and personal or
credential-like identifiers. None were found. The binary checkpoint was
inspected for top-level metadata only (`model_state`, `scalers`, `bands`,
`input_dim`) and contains no top-level source path field.
