# Canopy2Tuber — revised inference and field-training package

This package implements the revised MST-Net V3 LAI model and the frozen interval biomass calculations. It preserves a separate legacy inference entry point. Python 3.8.20, NumPy 1.24.3, pandas 2.0.3 and PyTorch 2.4.1 CPU were used for verification.

## Quick start
Run commands from this folder after installing `requirements.txt` in a suitable environment:
```console
python code/predict_lai.py --input example_data/synthetic_spectra.csv --output example_outputs/predicted_LAI.csv
python code/biomass_model.py --input example_data/synthetic_growth.csv --parameters config/historical_training383.json --output example_outputs/predicted_biomass.csv
python code/biomass_model.py --input example_data/synthetic_growth.csv --parameters config/external_historical550.json --external --output example_outputs/external_biomass.csv
```
Examples are synthetic and demonstrate execution only. Their outputs are not research accuracy estimates.

## LAI retrieval
Inputs are reflectance fractions (not percentages) in columns `450,560,650,730,840`; an optional ID column is retained. The model uses fixed PROSAIL-derived normalization, band embedding, three Transformer layers (128 features, four heads, FFN 256, default ReLU/post-normalization), attention pooling, and three residual CNN blocks (32/64/128 channels). Concatenated 256-dimensional features feed a 128/64/1 regression head. GELU is used in the residual CNN and regression head; there is no additional residual fusion connection.

Three final checkpoints and three PROSAIL-pretrained checkpoints are included (seeds 3407/3408/3409). Inference produces separate predictions for each run. The paper reports mean ± sample SD of run-specific metrics, not pooled or ensemble accuracy. Predictions are not clipped to the observed LAI range.

The field comparison uses 286 calibration observations (228 fitting + 58 internal selection) and 122 reserved evaluation observations, assigned at sample level. Cultivars occur in both sets; this is not a cultivar-held-out test. All settings are selected internally, followed by refitting on all 286 observations.

To repeat field selection and refitting after obtaining the calibration data:
```console
python code/train_full_v3.py --calibration calibration.csv --output trained_models
```
Required columns: the five wavelength columns, `lai`, and `Inner_split` with `Training`/`Selection`. Preserve the supplied assignments; do not randomly regenerate them. This entry point reproduces the full V3 selection/refit stage from supplied pretrained weights. PROSAIL LUT generation/pretraining from scratch, other architectures' training, and raw datasets are not included. The saved model comparison is not a new training run.

## Biomass calculations
`code/biomass_model.py` requires `Sample_ID,Date,N_Level (kg/ha),D,GDD_After_Max_LAI,APAR_Delta`, plus `Experiment` for historical prediction. `Sample_ID` identifies an entire trajectory globally; rows within it must be chronological. `Date` means accumulated thermal time (°Cd), not a calendar date. `GDD_After_Max_LAI` is the supplied nonnegative post-peak thermal age. `APAR_Delta` is intercepted PAR over each interval (MJ m−2), including the initial interval. No biomass observations are used for prediction.

The interval calculation is RUE = max[0, (base + alpha D)(1 − exp(−beta(N + 80)))exp(−delta(N + 80)) − gamma Age]; total dry biomass is the cumulative sum of RUE × interval APAR. HI is logistic, root fraction is exponential, and leaf/aboveground fraction is linear in thermal time. Tuber = total × HI; root = total × (1 − HI) × root fraction; aboveground = total × (1 − HI) × (1 − root fraction); leaf and stem partition aboveground biomass. All biomass predictions are dry matter in g m−2.

`historical_training383.json` contains coefficients fitted using the 383 historical training records. Missing treatment curves are interpolated over N within density, then over density, using only the stored curves. `external_historical550.json` contains six donor models fitted from all 550 historical records. The external mode averages their final biomass predictions with equal weights; it does not fit 2025 outcomes or average parameter vectors. These are distinct evaluation settings and parameter banks.

The all-data empirical RUE calibration table uses current RUE × cumulative APAR, a different calculation from this interval evaluation. Its coefficients must not be substituted into these interval banks. The old simplified biomass demo is therefore not the default executable workflow.

Retained historical LAI/APAR/thermal-time inputs remain fixed. The revised V3 retrieval comparison does not regenerate the biomass inputs. This package does not reconstruct weather integration, spatial maps, or raw canopy time series. If deriving fIPAR separately, the study uses 1 − exp(−0.5 LAI); interval APAR requires the corresponding radiation/time integration, not a reconstructed PAR proxy.

The six historical experiments span 2022–2024 and three cultivars; 2025 provides 75 observations from 15 plots of another cultivar. Historical split counts are 383/82/85 by whole sampling unit. D1/D2/D3 denote 74,100/55,575/44,460 plants ha−1 in the PP manuscript. The external evaluation does not isolate cultivar effects from season effects.

## Scope and verification
See `VERIFICATION.json` for numerical replay checks. Full field data are available from the corresponding author on reasonable request; see `DATA_AVAILABILITY.md`. Synthetic examples must not be used to calculate reported model performance. Fraction extrapolations outside [0,1] are flagged rather than silently clipped.

## License and citation
Canopy2Tuber is distributed under the [MIT License](LICENSE). This license covers the authors' source code, documentation, supplied model weights (including the legacy weights), fitted parameter files, and synthetic examples and their outputs in this package. Commercial use, modification and redistribution are permitted subject to the license terms. Third-party dependencies retain their respective licenses.

Complete raw phenotypic data are not included and are not covered by this license. They are available from the corresponding author upon reasonable request; see [Data availability](DATA_AVAILABILITY.md).

If you use Canopy2Tuber in your research, please cite the associated publication. This citation request is not an additional condition of the MIT License.
