# Review-preparation changes

This package was prepared for anonymous code sharing and reviewer inspection.

## Documentation added

- Added a root `README.md` with model purpose, installation, quick demo and full run order.
- Added `docs/REVIEWER_QUICK_START.md` for a five-replication demonstration run.
- Added `docs/GITHUB_UPLOAD_GUIDE.md` for anonymous GitHub upload steps.
- Added `docs/REPRODUCIBILITY_CHECKLIST.md`.
- Added `docs/MODEL_OVERVIEW.md`.
- Added `requirements.txt`, `environment.yml`, `CODE_AVAILABILITY.md`, `CITATION.cff` and `LICENSE`.

## Code-readability changes

- Expanded English module and function comments in the core ABM file.
- Added explanatory docstrings for household representation, spatial advantage,
  CE actions, recognition observability, and scenario execution.
- Replaced remaining Turkish comments and console messages with English.
- Added an optional quick-demo mode to `run_replicated_analysis.py`:
  `CE_ABM_QUICK_DEMO=1 python src/analysis/run_replicated_analysis.py`.

## Anonymization and cleanup

- Removed Python bytecode/cache files.
- Added `.gitignore` for generated outputs and temporary files.
- Replaced author metadata in Excel input files with anonymous metadata.
- Checked for obvious author names, emails and local file paths.
