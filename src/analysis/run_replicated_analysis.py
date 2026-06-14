#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_replicated_analysis.py

Main experiment runner for replicated Amsterdam CE-ABM scenarios.

This script executes the core scenarios repeatedly with different random seeds,
collects the monthly model outputs and writes all run metadata needed for
reproducibility. It also runs two small built-in sensitivity checks for the
reduced-form CE--unemployment coupling and the observability of CE actions.

Typical use:
1. Run this script from the repository root.
2. Use the generated run group to build figures with plotting/plot_replicated_figures.py.
3. Use the manifest and runtime tables in results/tables/ to inspect which runs
   were completed and which settings were used.

Set CE_ABM_QUICK_DEMO=1 to run a small five-replication demonstration instead
of the full manuscript-scale experiment.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import copy
import csv
import json
import os
import sys
import time
import traceback
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

# Make src importable and resolve project paths when this script is run from
# src/analysis.
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent
ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import abm as model  # noqa: E402


# =============================================================================
# User settings
# =============================================================================

BASE_SEED = 12345
LAST_WINDOW = 24

RUN_MAIN_REPLICATED_ANALYSIS = True
RUN_BETA_SENSITIVITY = True
RUN_OBSERVABILITY_SENSITIVITY = True

# -----------------------------------------------------------------------------
# Main replicated analysis: sequential stopping
# -----------------------------------------------------------------------------
USE_SEQUENTIAL_STOPPING = True
MIN_REPS = 10
MAX_REPS = 100
CHECK_EVERY = 5

# The stopping rule checks whether the relative 95% CI half-width of the
# last-window mean is below REL_CI_TOL for all selected core metrics/scenarios.
REL_CI_TOL = 0.05
# For near-zero means, use an absolute tolerance to avoid division instability.
REL_CI_MIN_DENOM = 1e-4
ABS_CI_TOL = 1e-4

# Used only if USE_SEQUENTIAL_STOPPING = False.
FIXED_N_REPS = 20

# Sensitivity analyses are intentionally smaller than the main experiment.
SENSITIVITY_N_REPS = 5

# Keep heavy per-cycle spatial PDFs off during replicated runs.
SAVE_SPATIAL_FIGURES = False
SPATIAL_FIGURE_INTERVAL = 12
SPATIAL_FIGURE_CYCLES = [0, 120, 240]

# Keep compact K1 spatial arrays on. This adds one compressed NPZ per run and
# allows plot_replicated_figures.py to build scenario heat maps without rerunning
# the simulations.
SAVE_K1_SPATIAL_DATA = True
K1_SPATIAL_LAST_WINDOW = LAST_WINDOW

SCENARIOS = [
    "ce_off",
    "ce_light",
    "baseline",
    "ce_strong",
]

SCENARIO_LABELS = {
    "ce_off": "Linear reference",
    "ce_light": "Low-intensity CE",
    "baseline": "Status-quo CE",
    "ce_strong": "Targeted CE expansion",
    "ce_inclusive_strong": "Inclusive CE expansion",
}

# Sensitivity for the reduced-form CE--unemployment coupling.
BETA_CEI_VALUES = [0.005, 0.0, -0.005, -0.015, -0.030]
BETA_SENSITIVITY_SCENARIOS = ["ce_light", "baseline", "ce_strong"]

# Sensitivity for the CE-action observability mechanism.
OBSERVABILITY_VALUES = [0.0, 0.5, 1.0]
OBSERVABILITY_SENSITIVITY_SCENARIOS = ["baseline", "ce_strong"]

# Metrics used for the sequential stopping rule. K1 is checked with its new
# explicit name; CE_INTENSITY remains available as a backward-compatible alias in
# abm.py outputs.
CONVERGENCE_METRICS = [
    "MEAN_LOCAL_CE_ACTIVATION",
    "CEI_SIM",
    "UNEMPLOYMENT",
    "D_adv_exposure",
    "D_resources",
    "MEAN_RESOURCES",
]

# Short run group prevents Windows path-length problems.
RUN_GROUP = datetime.now().strftime("r%Y%m%d_%H%M%S")

# Store original/default model parameters to avoid leakage between sensitivity runs.
DEFAULT_UNEMP_BETA_CEI = float(getattr(model, "BASE_UNEMP_BETA_CEI", getattr(model, "UNEMP_BETA_CEI", -0.015)))
DEFAULT_OBSERVABILITY_PROB = float(getattr(model, "BASE_OBSERVABILITY_PROB", getattr(model, "OBSERVABILITY_PROB", 0.5)))

# Optional quick-demo mode for reviewers. Run with:
#   CE_ABM_QUICK_DEMO=1 python src/analysis/run_replicated_analysis.py
# This does not change the manuscript settings below; it simply overrides them
# at runtime so that users can test the pipeline quickly on a laptop.
if os.environ.get("CE_ABM_QUICK_DEMO", "").strip() == "1":
    USE_SEQUENTIAL_STOPPING = False
    FIXED_N_REPS = 5
    RUN_BETA_SENSITIVITY = False
    RUN_OBSERVABILITY_SENSITIVITY = False
    SENSITIVITY_N_REPS = 2


# =============================================================================
# Paths
# =============================================================================

RESULTS_DIR = ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
TABLES_DIR = RESULTS_DIR / "tables"
EXPERIMENT_DIR = RESULTS_DIR / "experiments" / RUN_GROUP

for p in (RUNS_DIR, TABLES_DIR, EXPERIMENT_DIR):
    p.mkdir(parents=True, exist_ok=True)

MANIFEST_CSV = TABLES_DIR / f"{RUN_GROUP}_run_manifest.csv"
RUNTIME_BY_TASK_CSV = TABLES_DIR / f"{RUN_GROUP}_runtime_by_task.csv"
RUNTIME_SUMMARY_CSV = TABLES_DIR / f"{RUN_GROUP}_runtime_summary.csv"
CONFIG_JSON = TABLES_DIR / f"{RUN_GROUP}_analysis_config.json"
CONVERGENCE_CSV = TABLES_DIR / f"{RUN_GROUP}_convergence_diagnostics.csv"
STOPPING_JSON = TABLES_DIR / f"{RUN_GROUP}_stopping_decision.json"


# =============================================================================
# Helpers
# =============================================================================

def _json_safe(obj: Any) -> Any:
    """Convert objects that JSON cannot serialize by default."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _set_model_seed(seed: int) -> None:
    model.SIM_SEED = int(seed)
    model.random.seed(int(seed))
    model.np.random.seed(int(seed))


def _disable_spatial_plotting_if_requested() -> None:
    """
    Avoid spatial-figure overhead during large replicated experiments.

    This suppresses heavy PDF map exports. It does not suppress compact K1 spatial
    NPZ arrays, which are controlled separately by SAVE_K1_SPATIAL_DATA.
    """
    setattr(model, "SAVE_SPATIAL_FIGURES", bool(SAVE_SPATIAL_FIGURES))
    setattr(model, "SPATIAL_FIGURE_INTERVAL", int(SPATIAL_FIGURE_INTERVAL))
    setattr(model, "SPATIAL_FIGURE_CYCLES", list(SPATIAL_FIGURE_CYCLES))
    setattr(model, "SAVE_K1_SPATIAL_DATA", bool(SAVE_K1_SPATIAL_DATA))
    setattr(model, "K1_SPATIAL_LAST_WINDOW", int(K1_SPATIAL_LAST_WINDOW))

    if SAVE_SPATIAL_FIGURES:
        return

    def _noop(*args, **kwargs):
        return None

    candidates = [
        "plot_infrastructure",
        "plot_cell_mean",
        "plot_grid",
        "plot_spatial",
        "plot_spatial_map",
        "save_spatial_figures",
        "plot_cell_metric",
    ]
    for name in candidates:
        if hasattr(model, name):
            try:
                setattr(model, name, _noop)
                print(f"[INFO] Spatial plotting disabled by monkey patch: abm.{name}()")
            except Exception:
                pass


def _write_config_json() -> None:
    cfg = {
        "run_group": RUN_GROUP,
        "root": str(ROOT),
        "base_seed": BASE_SEED,
        "last_window": LAST_WINDOW,
        "run_main_replicated_analysis": RUN_MAIN_REPLICATED_ANALYSIS,
        "run_beta_sensitivity": RUN_BETA_SENSITIVITY,
        "run_observability_sensitivity": RUN_OBSERVABILITY_SENSITIVITY,
        "use_sequential_stopping": USE_SEQUENTIAL_STOPPING,
        "min_reps": MIN_REPS,
        "max_reps": MAX_REPS,
        "check_every": CHECK_EVERY,
        "fixed_n_reps_if_no_stopping": FIXED_N_REPS,
        "rel_ci_tol": REL_CI_TOL,
        "rel_ci_min_denom": REL_CI_MIN_DENOM,
        "abs_ci_tol": ABS_CI_TOL,
        "convergence_metrics": CONVERGENCE_METRICS,
        "sensitivity_n_reps": SENSITIVITY_N_REPS,
        "scenarios": SCENARIOS,
        "scenario_labels": SCENARIO_LABELS,
        "default_unemp_beta_cei": DEFAULT_UNEMP_BETA_CEI,
        "default_observability_prob": DEFAULT_OBSERVABILITY_PROB,
        "beta_cei_values": BETA_CEI_VALUES,
        "beta_sensitivity_scenarios": BETA_SENSITIVITY_SCENARIOS,
        "observability_values": OBSERVABILITY_VALUES,
        "observability_sensitivity_scenarios": OBSERVABILITY_SENSITIVITY_SCENARIOS,
        "save_spatial_figures": SAVE_SPATIAL_FIGURES,
        "spatial_figure_interval": SPATIAL_FIGURE_INTERVAL,
        "spatial_figure_cycles": SPATIAL_FIGURE_CYCLES,
        "save_k1_spatial_data": SAVE_K1_SPATIAL_DATA,
        "k1_spatial_last_window": K1_SPATIAL_LAST_WINDOW,
        "kpi_definitions": {
            "K1": "mean local CE activation; compact cell-level grids saved per run",
            "K2": "flow-based citywide CE index; excludes K1 to avoid double-counting",
        },
        "outputs": {
            "manifest_csv": str(MANIFEST_CSV),
            "runtime_by_task_csv": str(RUNTIME_BY_TASK_CSV),
            "runtime_summary_csv": str(RUNTIME_SUMMARY_CSV),
            "convergence_csv": str(CONVERGENCE_CSV),
            "stopping_json": str(STOPPING_JSON),
            "runs_dir": str(RUNS_DIR),
            "tables_dir": str(TABLES_DIR),
            "experiment_dir": str(EXPERIMENT_DIR),
        },
        "model_framing": "empirically grounded scenario-comparison model; not a calibrated point forecast",
        "replication_design": "sequential stopping for main scenarios; smaller fixed-replication one-at-a-time sensitivity analyses",
        "demographic_framing": "representative household decision units with household-size attribute; neighborhood-level population and income grounding, not full cell-specific demographic calibration",
    }
    CONFIG_JSON.write_text(json.dumps(cfg, indent=2, ensure_ascii=False, default=_json_safe), encoding="utf-8")


def _init_csvs() -> None:
    with open(MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "run_group", "experiment", "scenario", "scenario_label", "replicate", "seed",
            "beta_cei", "observability_prob", "run_tag", "run_dir", "master_csv",
            "k1_spatial_npz", "duration_sec", "status", "error",
        ])
        writer.writeheader()

    with open(RUNTIME_BY_TASK_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "run_group", "stage", "experiment", "scenario", "replicate", "seed",
            "beta_cei", "observability_prob", "duration_sec", "status",
        ])
        writer.writeheader()

    with open(CONVERGENCE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "run_group", "check_rep", "scenario", "metric", "n", "mean", "sd",
            "ci95_half_width", "relative_half_width", "passed", "criterion",
        ])
        writer.writeheader()


def _append_manifest(row: Dict[str, Any]) -> None:
    with open(MANIFEST_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "run_group", "experiment", "scenario", "scenario_label", "replicate", "seed",
            "beta_cei", "observability_prob", "run_tag", "run_dir", "master_csv",
            "k1_spatial_npz", "duration_sec", "status", "error",
        ])
        writer.writerow(row)


def _append_runtime(row: Dict[str, Any]) -> None:
    with open(RUNTIME_BY_TASK_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "run_group", "stage", "experiment", "scenario", "replicate", "seed",
            "beta_cei", "observability_prob", "duration_sec", "status",
        ])
        writer.writerow(row)


def _short_beta_label(beta: float) -> str:
    return f"b{beta:+.3f}".replace("+", "p").replace("-", "m").replace(".", "p")


def _short_obs_label(obs: float) -> str:
    return f"o{obs:.2f}".replace(".", "p")


def _get_model_run_dir() -> Path:
    rd = getattr(model, "RUN_DIR", None)
    if rd is not None:
        return Path(rd)
    return RUNS_DIR / str(getattr(model, "BASE_RUN_NAME", "unknown_run"))


def _build_policy_cfg(
    scenario_name: str,
    seed: int,
    rep: int,
    run_tag: str,
    beta_cei: Optional[float] = None,
    observability_prob: Optional[float] = None,
) -> Tuple[Dict[str, Any], float, float]:
    if scenario_name not in model.POLICY_SCENARIOS:
        raise KeyError(f"Scenario '{scenario_name}' not found in abm.POLICY_SCENARIOS.")

    effective_beta = DEFAULT_UNEMP_BETA_CEI if beta_cei is None else float(beta_cei)
    effective_obs = DEFAULT_OBSERVABILITY_PROB if observability_prob is None else float(observability_prob)

    policy_cfg = copy.deepcopy(model.POLICY_SCENARIOS[scenario_name])
    policy_cfg.update({
        "SEED": int(seed),
        "REPLICATE": int(rep),
        "RUN_TAG": str(run_tag),
        "_RUN_TAG": str(run_tag),
        "UNEMP_BETA_CEI": float(effective_beta),
        "OBSERVABILITY_PROB": float(effective_obs),
        "SAVE_SPATIAL_FIGURES": bool(SAVE_SPATIAL_FIGURES),
        "SPATIAL_FIGURE_INTERVAL": int(SPATIAL_FIGURE_INTERVAL),
        "SPATIAL_FIGURE_CYCLES": list(SPATIAL_FIGURE_CYCLES),
        "SAVE_K1_SPATIAL_DATA": bool(SAVE_K1_SPATIAL_DATA),
        "K1_SPATIAL_LAST_WINDOW": int(K1_SPATIAL_LAST_WINDOW),
    })

    return policy_cfg, effective_beta, effective_obs


def _run_single(
    experiment: str,
    scenario_name: str,
    rep: int,
    seed: int,
    run_tag: str,
    beta_cei: Optional[float] = None,
    observability_prob: Optional[float] = None,
) -> Optional[Dict[str, List[float]]]:
    policy_cfg, effective_beta, effective_obs = _build_policy_cfg(
        scenario_name=scenario_name,
        seed=seed,
        rep=rep,
        run_tag=run_tag,
        beta_cei=beta_cei,
        observability_prob=observability_prob,
    )

    print(
        f"\n[{experiment}:{scenario_name}] rep {rep + 1} | seed={seed} | "
        f"beta={effective_beta} | obs={effective_obs}"
    )

    t0 = time.perf_counter()
    status = "ok"
    error = ""
    run_dir = ""
    master_csv = ""
    k1_spatial_npz = ""
    log = None

    try:
        _set_model_seed(seed)
        model.BASE_RUN_NAME = f"{RUN_GROUP}_{run_tag}_r{rep:03d}"

        # Reset globals every run to avoid leakage between sensitivity settings.
        model.UNEMP_BETA_CEI = float(effective_beta)
        setattr(model, "OBSERVABILITY_PROB", float(effective_obs))
        setattr(model, "SAVE_SPATIAL_FIGURES", bool(SAVE_SPATIAL_FIGURES))
        setattr(model, "SPATIAL_FIGURE_INTERVAL", int(SPATIAL_FIGURE_INTERVAL))
        setattr(model, "SPATIAL_FIGURE_CYCLES", list(SPATIAL_FIGURE_CYCLES))
        setattr(model, "SAVE_K1_SPATIAL_DATA", bool(SAVE_K1_SPATIAL_DATA))
        setattr(model, "K1_SPATIAL_LAST_WINDOW", int(K1_SPATIAL_LAST_WINDOW))

        log = model.run_sim(scenario_name, policy_cfg)
        run_dir_path = _get_model_run_dir()
        run_dir = str(run_dir_path)
        master_path = run_dir_path / "MASTER_metrics.csv"
        k1_path = run_dir_path / "K1_spatial_ce_intensity.npz"
        master_csv = str(master_path if master_path.exists() else "")
        k1_spatial_npz = str(k1_path if k1_path.exists() else "")

    except Exception as exc:
        status = "failed"
        error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        print("[ERROR] Run failed:", error)
        tb_path = EXPERIMENT_DIR / f"error_{experiment}_{scenario_name}_r{rep:03d}.txt"
        tb_path.write_text(traceback.format_exc(), encoding="utf-8")
        raise

    finally:
        duration = time.perf_counter() - t0
        manifest_row = {
            "run_group": RUN_GROUP,
            "experiment": experiment,
            "scenario": scenario_name,
            "scenario_label": SCENARIO_LABELS.get(scenario_name, scenario_name),
            "replicate": rep,
            "seed": seed,
            "beta_cei": effective_beta,
            "observability_prob": effective_obs,
            "run_tag": run_tag,
            "run_dir": run_dir,
            "master_csv": master_csv,
            "k1_spatial_npz": k1_spatial_npz,
            "duration_sec": f"{duration:.6f}",
            "status": status,
            "error": error,
        }
        _append_manifest(manifest_row)
        _append_runtime({
            "run_group": RUN_GROUP,
            "stage": "simulation",
            "experiment": experiment,
            "scenario": scenario_name,
            "replicate": rep,
            "seed": seed,
            "beta_cei": effective_beta,
            "observability_prob": effective_obs,
            "duration_sec": f"{duration:.6f}",
            "status": status,
        })

    return log


# =============================================================================
# Sequential stopping helpers
# =============================================================================

def _last_window_mean(log: Dict[str, List[float]], metric: str) -> float:
    vals = log.get(metric, [])
    if vals is None or len(vals) == 0:
        return np.nan
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    w = min(LAST_WINDOW, arr.size)
    return float(np.nanmean(arr[-w:]))


def _ci_stats(values: List[float]) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "hw": np.nan, "rel_hw": np.nan, "passed": False, "criterion": "no_data"}
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    hw = float(1.96 * sd / np.sqrt(n)) if n > 1 else np.nan

    if n <= 1:
        return {"n": n, "mean": mean, "sd": sd, "hw": hw, "rel_hw": np.nan, "passed": False, "criterion": "n<=1"}

    denom = max(abs(mean), REL_CI_MIN_DENOM)
    rel_hw = float(hw / denom)

    if abs(mean) < REL_CI_MIN_DENOM:
        passed = bool(hw <= ABS_CI_TOL)
        criterion = f"abs_hw<={ABS_CI_TOL:g}"
    else:
        passed = bool(rel_hw <= REL_CI_TOL)
        criterion = f"rel_hw<={REL_CI_TOL:g}"

    return {"n": n, "mean": mean, "sd": sd, "hw": hw, "rel_hw": rel_hw, "passed": passed, "criterion": criterion}


def _evaluate_stopping(main_logs: Dict[str, List[Dict[str, List[float]]]], check_rep: int) -> Tuple[bool, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    all_passed = True

    for scenario_name in SCENARIOS:
        logs = main_logs.get(scenario_name, [])
        for metric in CONVERGENCE_METRICS:
            vals = [_last_window_mean(log, metric) for log in logs]
            stats = _ci_stats(vals)
            passed = bool(stats["passed"])
            if not passed:
                all_passed = False
            rows.append({
                "run_group": RUN_GROUP,
                "check_rep": check_rep,
                "scenario": scenario_name,
                "metric": metric,
                "n": stats["n"],
                "mean": stats["mean"],
                "sd": stats["sd"],
                "ci95_half_width": stats["hw"],
                "relative_half_width": stats["rel_hw"],
                "passed": passed,
                "criterion": stats["criterion"],
            })

    return all_passed, rows


def _append_convergence_rows(rows: List[Dict[str, Any]]) -> None:
    with open(CONVERGENCE_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "run_group", "check_rep", "scenario", "metric", "n", "mean", "sd",
            "ci95_half_width", "relative_half_width", "passed", "criterion",
        ])
        for row in rows:
            writer.writerow(row)


def _print_stopping_summary(check_rep: int, all_passed: bool, rows: List[Dict[str, Any]]) -> None:
    failed = [r for r in rows if not r["passed"]]
    print("\n------------------------------------------------------------")
    print(f"Sequential stopping check after {check_rep} replications")
    print(f"All criteria passed: {all_passed}")
    if failed:
        worst = sorted(
            failed,
            key=lambda r: np.nan_to_num(float(r["relative_half_width"]), nan=999.0),
            reverse=True,
        )[:5]
        print("Largest remaining relative half-widths:")
        for r in worst:
            print(
                f"  {r['scenario']} | {r['metric']} | "
                f"rel_hw={r['relative_half_width']:.4g} | hw={r['ci95_half_width']:.4g}"
            )
    print("------------------------------------------------------------\n")


# =============================================================================
# Experiment execution
# =============================================================================

def _run_main() -> Dict[str, List[Dict[str, List[float]]]]:
    main_logs: Dict[str, List[Dict[str, List[float]]]] = {s: [] for s in SCENARIOS}

    if not USE_SEQUENTIAL_STOPPING:
        for scenario_name in SCENARIOS:
            for rep in range(FIXED_N_REPS):
                seed = BASE_SEED + rep
                log = _run_single("main", scenario_name, rep, seed, run_tag="m")
                if log is not None:
                    main_logs[scenario_name].append(log)
        decision = {"mode": "fixed", "final_reps": FIXED_N_REPS, "stopped_by_criterion": False}
        STOPPING_JSON.write_text(json.dumps(decision, indent=2), encoding="utf-8")
        return main_logs

    stopped_by_criterion = False
    final_reps = MAX_REPS

    for rep in range(MAX_REPS):
        seed = BASE_SEED + rep
        print("\n============================================================")
        print(f"Main replicated analysis: replication {rep + 1}/{MAX_REPS}")
        print("============================================================")

        for scenario_name in SCENARIOS:
            log = _run_single("main", scenario_name, rep, seed, run_tag="m")
            if log is not None:
                main_logs[scenario_name].append(log)

        current_n = rep + 1
        should_check = current_n >= MIN_REPS and ((current_n - MIN_REPS) % CHECK_EVERY == 0 or current_n == MAX_REPS)
        if should_check:
            all_passed, rows = _evaluate_stopping(main_logs, current_n)
            _append_convergence_rows(rows)
            _print_stopping_summary(current_n, all_passed, rows)
            if all_passed:
                stopped_by_criterion = True
                final_reps = current_n
                break

    decision = {
        "mode": "sequential",
        "min_reps": MIN_REPS,
        "max_reps": MAX_REPS,
        "check_every": CHECK_EVERY,
        "final_reps": final_reps,
        "stopped_by_criterion": stopped_by_criterion,
        "rel_ci_tol": REL_CI_TOL,
        "abs_ci_tol": ABS_CI_TOL,
        "convergence_metrics": CONVERGENCE_METRICS,
    }
    STOPPING_JSON.write_text(json.dumps(decision, indent=2, ensure_ascii=False, default=_json_safe), encoding="utf-8")
    print(f"[STOPPING] Decision saved to: {STOPPING_JSON}")
    return main_logs


def _run_beta_sensitivity() -> None:
    for beta in BETA_CEI_VALUES:
        tag = _short_beta_label(beta)
        print("\n============================================================")
        print(f"Beta sensitivity: UNEMP_BETA_CEI = {beta}")
        print("============================================================")
        for scenario_name in BETA_SENSITIVITY_SCENARIOS:
            for rep in range(SENSITIVITY_N_REPS):
                seed = BASE_SEED + rep
                _run_single(
                    experiment="beta",
                    scenario_name=scenario_name,
                    rep=rep,
                    seed=seed,
                    run_tag=tag,
                    beta_cei=float(beta),
                    observability_prob=None,
                )


def _run_observability_sensitivity() -> None:
    for obs in OBSERVABILITY_VALUES:
        tag = _short_obs_label(obs)
        print("\n============================================================")
        print(f"Observability sensitivity: OBSERVABILITY_PROB = {obs}")
        print("============================================================")
        for scenario_name in OBSERVABILITY_SENSITIVITY_SCENARIOS:
            for rep in range(SENSITIVITY_N_REPS):
                seed = BASE_SEED + rep
                _run_single(
                    experiment="obs",
                    scenario_name=scenario_name,
                    rep=rep,
                    seed=seed,
                    run_tag=tag,
                    beta_cei=None,
                    observability_prob=float(obs),
                )


def _write_runtime_summary() -> None:
    rows: List[Dict[str, str]] = []
    with open(RUNTIME_BY_TASK_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    groups: Dict[tuple, List[float]] = {}
    for r in rows:
        key = (r["experiment"], r["scenario"])
        try:
            dur = float(r["duration_sec"])
        except Exception:
            continue
        groups.setdefault(key, []).append(dur)

    with open(RUNTIME_SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_group", "experiment", "scenario", "n_runs", "total_sec",
            "mean_sec", "min_sec", "max_sec",
        ])
        for (experiment, scenario), vals in sorted(groups.items()):
            n = len(vals)
            writer.writerow([
                RUN_GROUP,
                experiment,
                scenario,
                n,
                f"{sum(vals):.6f}",
                f"{(sum(vals) / n) if n else 0.0:.6f}",
                f"{min(vals):.6f}",
                f"{max(vals):.6f}",
            ])


def main() -> None:
    t0 = time.perf_counter()
    _disable_spatial_plotting_if_requested()
    _init_csvs()
    _write_config_json()

    print("============================================================")
    print("Replicated scenario execution")
    print("============================================================")
    print(f"Project root          : {ROOT}")
    print(f"Run group             : {RUN_GROUP}")
    print(f"Sequential stopping   : {USE_SEQUENTIAL_STOPPING}")
    
    if USE_SEQUENTIAL_STOPPING:
        print(f"Main reps             : min={MIN_REPS}, max={MAX_REPS}, check_every={CHECK_EVERY}")
    else:
        print(f"Main reps             : fixed={FIXED_N_REPS}")
    print(f"Sensitivity reps      : {SENSITIVITY_N_REPS}")
    print(f"Scenarios             : {', '.join(SCENARIOS)}")
    print(f"Default beta_CEI       : {DEFAULT_UNEMP_BETA_CEI}")
    print(f"Default observability  : {DEFAULT_OBSERVABILITY_PROB}")
    print(f"Spatial PDFs          : {SAVE_SPATIAL_FIGURES}")
    print(f"K1 spatial arrays     : {SAVE_K1_SPATIAL_DATA}")
    print(f"K1 last-window months : {K1_SPATIAL_LAST_WINDOW}")
    print(f"Manifest              : {MANIFEST_CSV}")
    print("============================================================")

    if RUN_MAIN_REPLICATED_ANALYSIS:
        _run_main()

    if RUN_BETA_SENSITIVITY:
        _run_beta_sensitivity()

    if RUN_OBSERVABILITY_SENSITIVITY:
        _run_observability_sensitivity()

    total_duration = time.perf_counter() - t0
    _append_runtime({
        "run_group": RUN_GROUP,
        "stage": "total",
        "experiment": "all",
        "scenario": "all",
        "replicate": "",
        "seed": "",
        "beta_cei": "",
        "observability_prob": "",
        "duration_sec": f"{total_duration:.6f}",
        "status": "ok",
    })
    _write_runtime_summary()

    print("\n[DONE] Replicated execution complete.")
    print(f"[DONE] Run group   : {RUN_GROUP}")
    print(f"[DONE] Manifest    : {MANIFEST_CSV}")
    print(f"[DONE] Runtime CSV : {RUNTIME_BY_TASK_CSV}")
    print(f"[DONE] Summary CSV : {RUNTIME_SUMMARY_CSV}")
    print(f"[DONE] Convergence : {CONVERGENCE_CSV}")
    print(f"[DONE] Stopping    : {STOPPING_JSON}")
    print("[NEXT] Build figures with:")
    print("       python src/plotting/plot_replicated_figures.py --run-group latest")


if __name__ == "__main__":
    main()
