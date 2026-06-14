# Run order for the Amsterdam CE-ABM code

This file explains the executable workflow. Run commands from the repository
root, not from inside `src/`.

## 0. Optional preprocessing

Only run preprocessing when raw data changed or when `data/processed/` is
missing. The repository already includes processed inputs for reviewer runs.

```bash
python src/preprocess_amsterdam.py
```

## 1. Fast reviewer/demo run

Use this to check that the pipeline works without waiting for the full analysis.
It runs five replications for the main scenarios and skips sensitivity runs.

```bash
CE_ABM_QUICK_DEMO=1 python src/analysis/run_replicated_analysis.py
```

Windows PowerShell:

```powershell
$env:CE_ABM_QUICK_DEMO="1"; python src/analysis/run_replicated_analysis.py
```

## 2. Full replicated scenario analysis

This runs the main replicated experiment and stores raw outputs, manifests,
runtime summaries and convergence diagnostics.

```bash
python src/analysis/run_replicated_analysis.py
```

Important output files:

```text
results/tables/rYYYYMMDD_HHMMSS_run_manifest.csv
results/tables/rYYYYMMDD_HHMMSS_convergence_diagnostics.csv
results/tables/rYYYYMMDD_HHMMSS_stopping_decision.json
results/experiments/rYYYYMMDD_HHMMSS/
```

## 3. Main manuscript figures

After the main run finishes, rebuild the main figures from saved outputs.

```bash
python src/plotting/plot_replicated_figures.py --run-group latest
```

Figures are written under `results/figures/<run_group>/`.

## 4. Verification and plausibility diagnostics

Run after the main replicated analysis and figure generation.

```bash
python src/analysis/run_verification_benchmarking.py --run-group latest
```

## 5. Extended sensitivity screening

Run the one-at-a-time sensitivity screening and reduced-mechanism benchmarks.

```bash
python src/analysis/run_extended_sensitivity_screening.py
```

## 6. Sensitivity figures

```bash
python src/plotting/plot_compact_sensitivity_figures.py --run-group latest
python src/plotting/plot_extended_sensitivity_figures_readable.py --run-group latest
```

## Notes

- Plotting scripts read stored CSV outputs and can be rerun without rerunning simulations.
- `latest` selects the newest compatible run group found in `results/tables/`.
- The main replicated analysis and extended sensitivity screening are separate runs.
- Large generated outputs are ignored by Git by default; add selected example outputs only if needed.
