# Amsterdam Circular-Economy Agent-Based Model (CE-ABM)

This repository contains the Python code and input data used to run an exploratory
agent-based scenario-comparison model of circular-economy (CE) practices,
spatial access and justice-oriented outcomes in Amsterdam.

The model is intended for **policy learning and scenario comparison**, not for
operational forecasting or exact infrastructure siting. It examines how
household CE actions, uneven access to CE infrastructure, recognition feedbacks,
accumulated economic resources and a reduced-form CE--unemployment coupling can
produce divergent circularity and social-equity outcomes.

## Main outputs

The code can reproduce the model workflow used in the manuscript:

- replicated scenario simulations;
- mean trajectories with 95% confidence intervals;
- last-window distributions and final-minus-initial change summaries;
- spatial diagnostics for mean local CE activation;
- verification and plausibility-benchmarking figures;
- extended one-at-a-time sensitivity and reduced-mechanism checks.

## Repository structure

```text
config/                  Scenario/configuration overrides
 data/
   raw/                  Raw Amsterdam spatial and socioeconomic input files
   external/             External data used by preprocessing utilities
   processed/            Preprocessed grid and initial household-agent files
results/                 Empty output folders; generated outputs are not tracked
src/
   abm.py                Core CE-ABM model
   preprocess_amsterdam.py
                         Builds processed grid and initial household files
   data_extraction.py    Optional utility for Repair Cafe data retrieval
   analysis/             Simulation and diagnostic runners
   plotting/             Figure-generation scripts
README.md                This file
requirements.txt         Python package requirements
LICENSE                  Draft open-source license for anonymous review
CITATION.cff             Draft citation metadata for anonymous review
```

## Installation

The code was developed for Python 3.10+ and uses standard scientific Python and
geospatial packages.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

`geopandas` and `rasterio` may require geospatial binary dependencies on some
systems. If pip installation fails, a conda environment is often easier:

```bash
conda create -n ce-abm python=3.11 numpy pandas matplotlib geopandas rasterio pyproj requests openpyxl -c conda-forge
conda activate ce-abm
```

## Quick test run

For a fast pipeline check, run a 5-replication demo of the main scenario set:

```bash
# macOS/Linux
CE_ABM_QUICK_DEMO=1 python src/analysis/run_replicated_analysis.py

# Windows PowerShell
$env:CE_ABM_QUICK_DEMO="1"; python src/analysis/run_replicated_analysis.py
```

Then create the main scenario figures from the newest run group:

```bash
python src/plotting/plot_replicated_figures.py --run-group latest
```

The demo is only for checking that the code runs. Manuscript figures should be
created from the full replicated analysis and sensitivity outputs.

## Full manuscript run order

Run all commands from the repository root.

### 1. Optional preprocessing

Only rerun preprocessing if raw data changed or `data/processed/` is missing:

```bash
python src/preprocess_amsterdam.py
```

### 2. Main replicated scenario analysis

```bash
python src/analysis/run_replicated_analysis.py
```

The main run uses up to 100 replications per scenario with sequential stopping.
Outputs are written to `results/runs/`, `results/experiments/` and
`results/tables/`.

### 3. Main manuscript figures

```bash
python src/plotting/plot_replicated_figures.py --run-group latest
```

### 4. Verification and plausibility diagnostics

```bash
python src/analysis/run_verification_benchmarking.py --run-group latest
```

### 5. Extended sensitivity screening

```bash
python src/analysis/run_extended_sensitivity_screening.py
```

### 6. Sensitivity figures

```bash
python src/plotting/plot_compact_sensitivity_figures.py --run-group latest
python src/plotting/plot_extended_sensitivity_figures_readable.py --run-group latest
```

More detailed execution notes are provided in `src/README_RUN_ORDER.md` and
`docs/REVIEWER_QUICK_START.md`.

## Key model concepts

- **K1: mean local CE activation**: spatial average of the grid-cell CE-activation field.
- **K2: flow-based citywide CE index**: weighted index of realized material drop-off flows and repair visits.
- **K3: CE flows**: sorting-container drop-off, bulky-waste drop-off and repair visits.
- **K4: unemployment**: aggregate unemployment rate under a reduced-form CE--unemployment coupling.
- **K5: normalized spatial-advantage exposure inequality**.
- **K6: accumulated economic-resource inequality**.

## Notes for anonymous peer review

This repository has been prepared without author names, affiliations or local
machine paths. If you use it during double-blind review, upload it from a neutral
GitHub account and keep the repository text anonymous until the review process is
complete.

After acceptance, replace the anonymous license/citation metadata with the final
author information and archive a release with a persistent identifier.


## Git LFS note

This full package includes the large `data/raw/bbga_kerncijfers.csv` file. The repository includes `.gitattributes` so this file is tracked with Git Large File Storage. Install and initialize Git LFS before committing/pushing the repository.
