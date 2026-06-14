# Reviewer quick start

This guide is for readers who want to check the model workflow without running
all manuscript-scale simulations.

## 1. Install packages

```bash
pip install -r requirements.txt
```

If geospatial package installation fails under pip, use conda:

```bash
conda create -n ce-abm python=3.11 numpy pandas matplotlib geopandas rasterio pyproj requests openpyxl -c conda-forge
conda activate ce-abm
```

## 2. Check that processed inputs exist

The repository includes preprocessed files under `data/processed/`:

- `amsterdam_grid.npz`
- `init_agents.pkl`
- `PREP_SUMMARY.json`

If these files are present, preprocessing is not needed for a quick run.

## 3. Run a 5-replication demo

```bash
CE_ABM_QUICK_DEMO=1 python src/analysis/run_replicated_analysis.py
```

Windows PowerShell:

```powershell
$env:CE_ABM_QUICK_DEMO="1"; python src/analysis/run_replicated_analysis.py
```

This run is deliberately small. It checks execution and produces example outputs,
but it is not the manuscript-scale analysis.

## 4. Build figures from the latest demo run

```bash
python src/plotting/plot_replicated_figures.py --run-group latest
```

Figures are written to `results/figures/<run_group>/`.

## 5. Full analysis

For manuscript-scale replication, run the scripts in the order given in
`README.md` or `src/README_RUN_ORDER.md`. The full analysis uses more
replications and can take longer.
