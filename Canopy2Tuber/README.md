# Canopy2Tuber

**An integrated UAV phenotyping, PROSAIL-informed deep learning, and dynamic
crop modeling framework for potato LAI retrieval and underground tuber yield
prediction.**

Canopy2Tuber is a minimal reproducibility package supporting transparent model
loading, LAI inference from five-band UAV spectra, and a representative dynamic
biomass workflow. It is designed to facilitate verification of the principal
computational pipeline while deliberately excluding complete UAV imagery, full
field datasets, training code, and non-deployment model checkpoints.

## Package structure

```text
Canopy2Tuber/
├── README.md
├── requirements.txt
├── LICENSE
├── DATA_AVAILABILITY.md
├── weights/MSTNet_LAI_best.pth
├── config/MSTNet_config.yaml
├── code/model.py
├── code/predict_lai.py
├── code/biomass_model_demo.py
├── code/utils.py
├── example_data/
└── example_outputs/
```

## Installation

Create a clean Python environment and install the minimal dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The package was tested with Python 3.8.20, PyTorch 2.4.1+cpu, NumPy 1.24.3,
and pandas 2.0.3. CPU inference is sufficient.

## LAI input format and band order

The input must be a CSV file with at least the following columns. Column names
are case-sensitive.

```text
Sample_ID,Blue,Green,Red,Red_edge,NIR
```

The required band order is **Blue (450 nm), Green (560 nm), Red (650 nm),
Red edge (730 nm), NIR (840 nm)**. Values must be finite numeric reflectance
values. If `Measured_LAI` is present, the script also writes R2, RMSE, and MAE
to the output file.

The model uses the training-set `StandardScaler` values stored in
`config/MSTNet_config.yaml`; the example data are never used to refit or
recalculate normalization.

## Run LAI inference

Run commands from the package root:

```bash
python code/predict_lai.py \
  --input example_data/example_UAV_spectra.csv \
  --weights weights/MSTNet_LAI_best.pth \
  --config config/MSTNet_config.yaml \
  --output example_outputs/predicted_LAI.csv
```

The checkpoint is loaded with `strict=True`. The script checks its input
dimension and band list before prediction.

## Run the biomass demonstration

```bash
python code/biomass_model_demo.py \
  --input example_data/example_growth_inputs.csv \
  --output example_outputs/predicted_biomass.csv
```

The demonstration uses one anonymized representative treatment sequence. It
replays `LAI -> fIPAR -> APAR -> dynamic RUE -> total biomass -> leaf, stem,
root, and tuber biomass` with source-derived parameters and allocation
fractions. `Incident_PAR_proxy` is algebraically derived from the source
APAR and fIPAR values because the original daily incident-PAR record is not
included in this minimal package; it is provided solely to make the example
calculation self-contained.

## Example data

- `example_UAV_spectra.csv`: 40 anonymized field spectra selected with a fixed
  seed (3407). Eight samples are drawn from each of five measured-LAI bins
  after applying an absolute prediction-error eligibility limit of 0.85 LAI.
  The resulting example range is 1.03--6.40 LAI. This is an intentionally
  filtered code-testing set, not a random validation set or a claim of overall
  field-model performance.
- `example_PROSAIL_spectra.csv`: 200 fixed-seed samples from the source
  PROSAIL simulation table. It illustrates the simulated spectral distribution
  and input format; it is not the full simulation dataset.
- `example_growth_inputs.csv`: four time points from one anonymized,
  representative model sequence. It has no plot coordinates, GPS data,
  personal information, or raw image data.

These small files are intended only to test installation and code execution;
they do not represent the full study dataset. In particular, the example UAV
file was selected to provide stable, broad-range demonstration predictions and
must not be used to report manuscript model accuracy.

## Expected outputs

- `example_outputs/predicted_LAI.csv`: Sample ID, LAI prediction, and (when
  available) global evaluation metrics for the supplied example file.
- `example_outputs/predicted_biomass.csv`: thermal time, LAI, fIPAR, APAR,
  RUE, total biomass, and leaf/stem/root/tuber biomass.

## Common issues

- **Missing columns**: Use exactly `Sample_ID, Blue, Green, Red, Red_edge,
  NIR`; do not use wavelength numbers as headers for this inference script.
- **Checkpoint mismatch**: Do not alter `model.py`, `MSTNet_config.yaml`, or
  band order when using the supplied weight file.
- **Unexpected reflectance scale**: Input spectra must use the same reflectance
  scale as the examples and source training data, not 0–255 digital numbers.
- **Different PyTorch version**: A newer compatible PyTorch release may work,
  but use the tested version for the most direct reproducibility check.

## Use restrictions

See `DATA_AVAILABILITY.md` and `LICENSE`. Authors must select a formal license
before public release. Full data may be available from the corresponding author
subject to institutional approval.
