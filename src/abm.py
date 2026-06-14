#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
abm.py

Core simulation engine for the Amsterdam circular-economy agent-based model.

This file defines the model state and monthly update logic used by all analysis
scripts. It loads the preprocessed Amsterdam grid and initial household agents,
constructs policy scenarios, simulates household CE decisions and records the
monthly indicators used in the paper.

Main responsibilities:
- load processed spatial and household inputs from data/processed/;
- define model-wide parameters and policy-scenario settings;
- represent households, repair hubs and the municipality as model objects;
- update household waste, sorting, bulky drop-off and repair decisions;
- update social recognition, spatial exposure and accumulated resources;
- aggregate monthly CE, unemployment and inequality indicators;
- save per-run CSV metrics and compact spatial arrays under results/.

The model is intended for scenario comparison and policy learning. It is not an
operational forecasting system for exact infrastructure placement.
"""
from pathlib import Path
from collections import defaultdict
import numpy as np
import random, pickle, warnings, csv, math, json, os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from datetime import datetime

# ---- Windows unicode guard
try:
    import sys
    if getattr(sys.stdout, "reconfigure", None):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if getattr(sys.stderr, "reconfigure", None):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ---- Figure style
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 15,
    "axes.titlesize": 22,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
})

# ---------------- Paths ----------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"
CONFIG_DIR = ROOT / "config"
RESULTS_DIR = ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
for _p in (RUNS_DIR, FIGURES_DIR, TABLES_DIR):
    _p.mkdir(parents=True, exist_ok=True)

GRID_FILE = DATA_PROCESSED / "amsterdam_grid.npz"
AGNT_FILE = DATA_PROCESSED / "init_agents.pkl"
SCEN_FILE = CONFIG_DIR / "scenario.json"

BASE_RUN_NAME = datetime.now().strftime("run_%Y%m%d_%H%M%S")
RUN_DIR = None

# ---------------- Defaults ----------------
GRID = 60
CYCLES = 240
GEN_STEP = 12

MAX_PATCH_CAPACITY = 12
NEED = 3.0
EXCLUSION_THRESHOLD = 1.0

HUB_TRADES_AS_WEALTH_BONUS = 0.20
SEC_TRADE_MULT             = 1.2
UNIT_TO_KG                 = 1.0

GAMMA_NEED_SAVINGS = 0.60

BULKY_SHARE     = 0.15
BULKY_THRESHOLD = 4.0
BULKY_CE_MULT   = 2.0

REPRO_THRESHOLD   = 150.0
REPRO_SHARE       = 0.30
REPRO_PROB        = 0.006
REPRO_MIN_AGE     = 22.0
REPRO_MAX_AGE     = 40.0

W_DIST_HUB       = 0.25
W_WASTE_EXP      = 2.2
W_RECOG_BON      = 1.1
W_WEALTH_INERTIA = 0.80
BETA_RECOG       = 0.30

MOVE_PROB        = 0.10

USE_BIOLOGICAL_DEMOGRAPHY = False
USE_HOUSEHOLD_SIZE_DYNAMICS = True
HOUSEHOLD_SIZE_MIN = 1
HOUSEHOLD_SIZE_MAX = 6
HOUSEHOLD_SIZE_REF = 2.0
HOUSEHOLD_SIZE_NEED_ELASTICITY = 0.75
HOUSEHOLD_SIZE_CHANGE_PROB = 0.002
HOUSEHOLD_SIZE_UP_PROB = 0.50

RESET_INITIAL_RECOGNITION = True
INITIAL_RECOG_ALPHA = 2.0
INITIAL_RECOG_BETA = 5.0

OBSERVABILITY_PROB = 0.50
W_RECOG_ACTION = 0.60

ENV_WEIGHT    = 0.75
SOCIAL_WEIGHT = 0.60
FIN_WEIGHT    = 0.20
MOTIVE_CONC   = 3.0

W_SEC_HUB = 0.80

SORT_SEGMENT_SHARES   = [0.25, 0.25, 0.30, 0.20]
SEG_SORT_TIME_MINUTES = [20.0, 30.0, 30.0, 90.0]
SEG_TIME_SENSITIVITY  = [1.3, 1.1, 0.7, 1.1]
SEG_CAPACITY_WEIGHT   = [0.6, 1.0, 1.3, 1.0]

SORT_TIME_VALUE_MIN = 2.8
SORT_TIME_VALUE_MAX = 6.3
GAMMA_TIME_COST     = 0.15

ALPHA_MOT = 1.0
ALPHA_CAP = 1.0
ALPHA_OPP = 1.0
BETA_COST  = 1.0
BETA_TIME  = 1.0
BETA_SKILL = 0.7
BETA_STIG  = 0.5

ETA_KNOW   = 0.05
ETA_OPP    = 0.05
ETA_INC    = 0.10

TRADE_RANGE = 3

SEASON_REGEN_DEF = [1.0] * 12
SEASON_TRADE_DEF = [1.0] * 12

ADV_HUB_WEIGHT      = 1.0
ADV_SEC_BONUS       = 0.5
ADV_WASTE_PENALTY   = 1.2
ADV_CONT_WEIGHT     = 0.6
ADV_RECYCL_WEIGHT   = 0.5

LAMBDA_PLUS  = 0.10
LAMBDA_MINUS = 0.05
CE_MEMORY    = 0.80

# Compact spatial data for K1 diagnostics. This saves one compressed NPZ per
# run, not monthly PDFs. It is used by plot_replicated_figures.py to build
# scenario-level heat maps of local CE activation, cell-level variability and
# final-minus-initial spatial change.
SAVE_K1_SPATIAL_DATA = True
K1_SPATIAL_LAST_WINDOW = 24

BASE_PRIMARY_WAGE   = 1.0
ALPHA_PRIMARY_DROP  = 0.5
PRIMARY_WAGE_FLOOR  = 0.2
BASE_CE_WAGE        = 0.8
CE_WAGE_SLOPE       = 1.0

# Flow-based CEI_sim -- unemployment coupling parameters.
# K2 is now a citywide flow-based CE index. It combines realized material
# drop-off flows and repair visits, but excludes CE_INTENSITY to avoid
# double-counting because CE_INTENSITY is itself derived from local CE flows.
JOB_CE_SENSITIVITY   = 0.12   # retained as metadata; not directly used
JOB_DISP_SENSITIVITY = 0.06   # retained as metadata; not directly used
CEI_W_TRADE          = 0.60   # weight for delivered CE material flows
CEI_W_REPAIR         = 0.40   # weight for repair visits
CEI_W_INT            = 0.0    # retained for backward compatibility; excluded from flow-based K2

UNEMP_REF      = 0.10
UNEMP_BETA_CEI = -0.015
UNEMP_MAX      = 0.35
CEI_REF        = 0.0
RISK_UNEMP_MULT = 1.3

SIM_SEED = 7

AGE_DIST_BINS = np.array([0.0, 15.0, 25.0, 45.0, 65.0, 80.0, 100.0], dtype=float)
AGE_DIST_WEIGHTS = np.array([0.16, 0.12, 0.28, 0.24, 0.14, 0.06], dtype=float)
AGE_DIST_WEIGHTS = AGE_DIST_WEIGHTS / AGE_DIST_WEIGHTS.sum()

def sample_initial_age():
    idx = np.random.choice(len(AGE_DIST_WEIGHTS), p=AGE_DIST_WEIGHTS)
    a0 = AGE_DIST_BINS[idx]
    a1 = AGE_DIST_BINS[idx + 1]
    return float(np.random.uniform(a0, a1))

def monthly_death_prob(age_years: float) -> float:
    a = float(age_years)
    if a < 1.0:
        p_y = 0.0038
    elif a < 5.0:
        p_y = 0.0004
    elif a < 15.0:
        p_y = 0.0001
    elif a < 25.0:
        p_y = 0.0002
    elif a < 45.0:
        p_y = 0.0005
    elif a < 65.0:
        p_y = 0.0030
    elif a < 80.0:
        p_y = 0.0120
    else:
        p_y = 0.0600
    return 1.0 - (1.0 - p_y) ** (1.0 / 12.0)


def sample_household_size() -> int:
    sizes = np.arange(HOUSEHOLD_SIZE_MIN, HOUSEHOLD_SIZE_MAX + 1, dtype=int)
    probs = np.array([0.38, 0.34, 0.13, 0.09, 0.04, 0.02], dtype=float)
    if probs.size != sizes.size:
        probs = np.ones_like(sizes, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(sizes, p=probs))


def household_size_factor(size: float) -> float:
    s = float(np.clip(size, HOUSEHOLD_SIZE_MIN, HOUSEHOLD_SIZE_MAX))
    ref = max(1e-9, float(HOUSEHOLD_SIZE_REF))
    return float((s / ref) ** HOUSEHOLD_SIZE_NEED_ELASTICITY)


def neutral_initial_recognition() -> float:
    return float(np.clip(np.random.beta(INITIAL_RECOG_ALPHA, INITIAL_RECOG_BETA), 0.0, 1.0))

CE_ACTIVE = True
SIG_CLIP = 50.0

def _normalize_12(vec, fallback):
    try:
        v = np.array(vec, dtype=float).flatten()
        if v.size != 12 or not np.all(np.isfinite(v)):
            raise ValueError
    except Exception:
        v = np.array(fallback, dtype=float)
    m = float(v.mean()) if float(v.mean()) > 1e-12 else 1.0
    return (v / m).tolist()

def _stable_sigmoid(x: float) -> float:
    x = float(np.clip(x, -SIG_CLIP, SIG_CLIP))
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

# ------------- Scenario overrides (scenario.json) -------------
def _get_overrides():
    if SCEN_FILE.exists():
        try:
            return json.loads(SCEN_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

sc = _get_overrides()
def getp(key, default):
    return sc.get(key, default)

CYCLES        = int(getp("CYCLES", CYCLES))
GEN_STEP      = int(getp("GEN_STEP", GEN_STEP))
MAX_PATCH_CAPACITY = int(getp("MAX_PATCH_CAPACITY", MAX_PATCH_CAPACITY))

NEED                = float(getp("NEED", NEED))
HUB_TRADES_AS_WEALTH_BONUS = float(getp("HUB_TRADES_AS_WEALTH_BONUS", HUB_TRADES_AS_WEALTH_BONUS))
EXCLUSION_THRESHOLD = float(getp("EXCLUSION_THRESHOLD", EXCLUSION_THRESHOLD))
GAMMA_NEED_SAVINGS  = float(getp("GAMMA_NEED_SAVINGS", GAMMA_NEED_SAVINGS))

BULKY_SHARE     = float(getp("BULKY_SHARE", BULKY_SHARE))
BULKY_THRESHOLD = float(getp("BULKY_THRESHOLD", BULKY_THRESHOLD))
BULKY_CE_MULT   = float(getp("BULKY_CE_MULT", BULKY_CE_MULT))

REPRO_THRESHOLD   = float(getp("REPRO_THRESHOLD", REPRO_THRESHOLD))
REPRO_SHARE       = float(getp("REPRO_SHARE", REPRO_SHARE))
REPRO_PROB        = float(getp("REPRO_PROB", REPRO_PROB))
REPRO_MIN_AGE     = float(getp("REPRO_MIN_AGE", REPRO_MIN_AGE))
REPRO_MAX_AGE     = float(getp("REPRO_MAX_AGE", REPRO_MAX_AGE))

W_DIST_HUB       = float(getp("W_DIST_HUB", W_DIST_HUB))
W_WASTE_EXP      = float(getp("W_WASTE_EXP", W_WASTE_EXP))
W_RECOG_BON      = float(getp("W_RECOG_BON", W_RECOG_BON))
W_WEALTH_INERTIA = float(getp("W_WEALTH_INERTIA", W_WEALTH_INERTIA))
BETA_RECOG       = float(getp("BETA_RECOG", BETA_RECOG))
MOVE_PROB        = float(getp("MOVE_PROB", MOVE_PROB))
MOVE_PROB        = float(np.clip(MOVE_PROB, 0.0, 1.0))

USE_BIOLOGICAL_DEMOGRAPHY = bool(getp("USE_BIOLOGICAL_DEMOGRAPHY", USE_BIOLOGICAL_DEMOGRAPHY))
USE_HOUSEHOLD_SIZE_DYNAMICS = bool(getp("USE_HOUSEHOLD_SIZE_DYNAMICS", USE_HOUSEHOLD_SIZE_DYNAMICS))
HOUSEHOLD_SIZE_MIN = int(getp("HOUSEHOLD_SIZE_MIN", HOUSEHOLD_SIZE_MIN))
HOUSEHOLD_SIZE_MAX = int(getp("HOUSEHOLD_SIZE_MAX", HOUSEHOLD_SIZE_MAX))
HOUSEHOLD_SIZE_REF = float(getp("HOUSEHOLD_SIZE_REF", HOUSEHOLD_SIZE_REF))
HOUSEHOLD_SIZE_NEED_ELASTICITY = float(getp("HOUSEHOLD_SIZE_NEED_ELASTICITY", HOUSEHOLD_SIZE_NEED_ELASTICITY))
HOUSEHOLD_SIZE_CHANGE_PROB = float(getp("HOUSEHOLD_SIZE_CHANGE_PROB", HOUSEHOLD_SIZE_CHANGE_PROB))
HOUSEHOLD_SIZE_CHANGE_PROB = float(np.clip(HOUSEHOLD_SIZE_CHANGE_PROB, 0.0, 1.0))
HOUSEHOLD_SIZE_UP_PROB = float(getp("HOUSEHOLD_SIZE_UP_PROB", HOUSEHOLD_SIZE_UP_PROB))
HOUSEHOLD_SIZE_UP_PROB = float(np.clip(HOUSEHOLD_SIZE_UP_PROB, 0.0, 1.0))
RESET_INITIAL_RECOGNITION = bool(getp("RESET_INITIAL_RECOGNITION", RESET_INITIAL_RECOGNITION))
INITIAL_RECOG_ALPHA = float(getp("INITIAL_RECOG_ALPHA", INITIAL_RECOG_ALPHA))
INITIAL_RECOG_BETA = float(getp("INITIAL_RECOG_BETA", INITIAL_RECOG_BETA))
OBSERVABILITY_PROB = float(getp("OBSERVABILITY_PROB", OBSERVABILITY_PROB))
OBSERVABILITY_PROB = float(np.clip(OBSERVABILITY_PROB, 0.0, 1.0))
W_RECOG_ACTION = float(getp("W_RECOG_ACTION", W_RECOG_ACTION))
W_RECOG_ACTION = float(max(0.0, W_RECOG_ACTION))

ENV_WEIGHT    = float(getp("ENV_WEIGHT", ENV_WEIGHT))
SOCIAL_WEIGHT = float(getp("SOCIAL_WEIGHT", SOCIAL_WEIGHT))
FIN_WEIGHT    = float(getp("FIN_WEIGHT", FIN_WEIGHT))
MOTIVE_CONC   = float(getp("MOTIVE_CONC", MOTIVE_CONC))
W_SEC_HUB     = float(getp("W_SEC_HUB", W_SEC_HUB))

ALPHA_MOT = float(getp("ALPHA_MOT", ALPHA_MOT))
ALPHA_CAP = float(getp("ALPHA_CAP", ALPHA_CAP))
ALPHA_OPP = float(getp("ALPHA_OPP", ALPHA_OPP))
BETA_COST  = float(getp("BETA_COST", BETA_COST))
BETA_TIME  = float(getp("BETA_TIME", BETA_TIME))
BETA_SKILL = float(getp("BETA_SKILL", BETA_SKILL))
BETA_STIG  = float(getp("BETA_STIG", BETA_STIG))

SEC_TRADE_MULT = float(getp("SEC_TRADE_MULT", SEC_TRADE_MULT))
UNIT_TO_KG     = float(getp("UNIT_TO_KG", UNIT_TO_KG)) or 1.0
TRADE_RANGE    = int(getp("TRADE_RANGE", TRADE_RANGE))
SIM_SEED       = int(getp("SEED", SIM_SEED))

SEASON_REGEN = _normalize_12(getp("SEASON_REGEN", SEASON_REGEN_DEF), SEASON_REGEN_DEF)
SEASON_TRADE = _normalize_12(getp("SEASON_TRADE", SEASON_TRADE_DEF), SEASON_TRADE_DEF)

ETA_KNOW   = float(getp("ETA_KNOW", ETA_KNOW))
ETA_OPP    = float(getp("ETA_OPP", ETA_OPP))
ETA_INC    = float(getp("ETA_INC", ETA_INC))

LAMBDA_PLUS  = float(getp("LAMBDA_PLUS", LAMBDA_PLUS))
LAMBDA_MINUS = float(getp("LAMBDA_MINUS", LAMBDA_MINUS))
CE_MEMORY    = float(getp("CE_MEMORY", CE_MEMORY))
SAVE_K1_SPATIAL_DATA = bool(getp("SAVE_K1_SPATIAL_DATA", SAVE_K1_SPATIAL_DATA))
K1_SPATIAL_LAST_WINDOW = int(getp("K1_SPATIAL_LAST_WINDOW", K1_SPATIAL_LAST_WINDOW))
K1_SPATIAL_LAST_WINDOW = max(1, K1_SPATIAL_LAST_WINDOW)

BASE_PRIMARY_WAGE  = float(getp("BASE_PRIMARY_WAGE", BASE_PRIMARY_WAGE))
ALPHA_PRIMARY_DROP = float(getp("ALPHA_PRIMARY_DROP", ALPHA_PRIMARY_DROP))
PRIMARY_WAGE_FLOOR = float(getp("PRIMARY_WAGE_FLOOR", PRIMARY_WAGE_FLOOR))
BASE_CE_WAGE       = float(getp("BASE_CE_WAGE", BASE_CE_WAGE))
CE_WAGE_SLOPE      = float(getp("CE_WAGE_SLOPE", CE_WAGE_SLOPE))

JOB_CE_SENSITIVITY   = float(getp("JOB_CE_SENSITIVITY", JOB_CE_SENSITIVITY))
JOB_DISP_SENSITIVITY = float(getp("JOB_DISP_SENSITIVITY", JOB_DISP_SENSITIVITY))
CEI_W_TRADE          = float(getp("CEI_W_TRADE", CEI_W_TRADE))
CEI_W_REPAIR         = float(getp("CEI_W_REPAIR", CEI_W_REPAIR))
# CE_INTENSITY is intentionally excluded from the flow-based K2 definition.
# We force CEI_W_INT to zero even if an older scenario.json contains CEI_W_INT.
CEI_W_INT            = 0.0

UNEMP_REF       = float(getp("UNEMP_REF", UNEMP_REF))
UNEMP_BETA_CEI  = float(getp("UNEMP_BETA_CEI", UNEMP_BETA_CEI))
UNEMP_MAX       = float(getp("UNEMP_MAX", UNEMP_MAX))
CEI_REF         = float(getp("CEI_REF", CEI_REF))
RISK_UNEMP_MULT = float(getp("RISK_UNEMP_MULT", RISK_UNEMP_MULT))

N_HOUSEHOLDS_SCALE = float(getp("N_HOUSEHOLDS_SCALE", 1.0))
HUB_ACTIVE_FRACTION = float(getp("HUB_ACTIVE_FRACTION", 1.0))
HUB_ACTIVE_FRACTION = max(0.0, min(HUB_ACTIVE_FRACTION, 1.0))
NEW_HUB_COUNT = int(getp("NEW_HUB_COUNT", 0))
NEW_HUB_NEAR_SECONDARY = bool(getp("NEW_HUB_NEAR_SECONDARY", False))

random.seed(SIM_SEED)
np.random.seed(SIM_SEED)
warnings.filterwarnings("ignore", category=RuntimeWarning)

EMPTY, PRIMARY, SECONDARY, WASTE, HUB = 0, 1, 2, 3, 4

# ---- Baseline parameter snapshot used to reset values between policy scenarios ----
BASE_ETA_KNOW   = ETA_KNOW
BASE_ETA_OPP    = ETA_OPP
BASE_ETA_INC    = ETA_INC
BASE_N_HOUSEHOLDS_SCALE  = N_HOUSEHOLDS_SCALE
BASE_HUB_ACTIVE_FRACTION = HUB_ACTIVE_FRACTION
BASE_NEW_HUB_COUNT       = NEW_HUB_COUNT
BASE_NEW_HUB_NEAR_SECONDARY = NEW_HUB_NEAR_SECONDARY
BASE_SEC_TRADE_MULT      = SEC_TRADE_MULT
BASE_HUB_TRADES_AS_WEALTH_BONUS = HUB_TRADES_AS_WEALTH_BONUS
BASE_LAMBDA_PLUS         = LAMBDA_PLUS
BASE_LAMBDA_MINUS        = LAMBDA_MINUS
BASE_GAMMA_NEED_SAVINGS  = GAMMA_NEED_SAVINGS
BASE_MOVE_PROB            = MOVE_PROB
BASE_SAVE_K1_SPATIAL_DATA  = SAVE_K1_SPATIAL_DATA
BASE_K1_SPATIAL_LAST_WINDOW = K1_SPATIAL_LAST_WINDOW
BASE_UNEMP_BETA_CEI       = UNEMP_BETA_CEI
BASE_RISK_UNEMP_MULT      = RISK_UNEMP_MULT
BASE_OBSERVABILITY_PROB   = OBSERVABILITY_PROB
BASE_W_RECOG_ACTION       = W_RECOG_ACTION

# ---------------- Load grid (base) ----------------
data = np.load(GRID_FILE, allow_pickle=True)
ptype_base      = data["ptype"]
stock_base      = data["stock"].astype(float)
regen_base_base = data["regen"].astype(float)
GRID0           = ptype_base.shape[0]

hub_cells_base       = [tuple(x) for x in data["hub_cells"]]
waste_cells_base     = [tuple(x) for x in data["waste_cells"]]
container_cells_base = [tuple(x) for x in (data["container_cells"] if "container_cells" in data.files else [])]
recycling_cells_base = [tuple(x) for x in (data["recycling_cells"] if "recycling_cells" in data.files else [])]

hub_fraction_base = {}
if "hub_fracs" in data:
    for row in data["hub_fracs"]:
        x, y, fr = row
        hub_fraction_base[(int(x), int(y))] = str(fr)
else:
    for loc in hub_cells_base:
        hub_fraction_base[loc] = "RepairCafe"

ptype = None
stock = None
regen_base = None
regen = None
ce_intensity = None
ce_flow = None
hub_cells = None
waste_cells = None
container_cells = None
recycling_cells = None
hub_fraction = None
GRID = GRID0

dist_hub       = None
dist_waste     = None
dist_container = None
dist_recycling = None

advantage_raw_field  = None
advantage_norm_field = None
curr_unemployment = UNEMP_REF

# ---------------- Policy scenario definitions ----------------
POLICY_SCENARIOS = {
    "ce_off": {
        "CE_ACTIVE": False,
        "ETA_KNOW": 0.0,
        "ETA_OPP":  0.0,
        "ETA_INC":  0.0,
    },
    "ce_light": {
        "CE_ACTIVE": True,
        "ETA_KNOW": BASE_ETA_KNOW * 0.3,
        "ETA_OPP":  BASE_ETA_OPP  * 0.3,
        "ETA_INC":  BASE_ETA_INC  * 0.3,
        "HUB_ACTIVE_FRACTION": min(1.0, BASE_HUB_ACTIVE_FRACTION * 0.5),
        "SEC_TRADE_MULT":      BASE_SEC_TRADE_MULT * 0.7,
        "HUB_TRADES_AS_WEALTH_BONUS": BASE_HUB_TRADES_AS_WEALTH_BONUS * 0.7,
        "GAMMA_NEED_SAVINGS":  BASE_GAMMA_NEED_SAVINGS * 0.7,
    },
    "baseline": {
    },
    "ce_strong": {
        "CE_ACTIVE": True,
        "ETA_KNOW": BASE_ETA_KNOW + 0.15,
        "ETA_OPP":  BASE_ETA_OPP  + 0.20,
        "ETA_INC":  BASE_ETA_INC  + 0.15,
        "HUB_ACTIVE_FRACTION": 1.0,
        "NEW_HUB_COUNT":       max(BASE_NEW_HUB_COUNT, 15),
        "NEW_HUB_NEAR_SECONDARY": True,
        "HUB_TRADES_AS_WEALTH_BONUS": BASE_HUB_TRADES_AS_WEALTH_BONUS * 2.0,
        "SEC_TRADE_MULT":      BASE_SEC_TRADE_MULT * 2.0,
        "GAMMA_NEED_SAVINGS":  BASE_GAMMA_NEED_SAVINGS * 1.3,
    },
    "ce_inclusive_strong": {
        "CE_ACTIVE": True,
        "ETA_KNOW": BASE_ETA_KNOW + 0.20,
        "ETA_OPP":  BASE_ETA_OPP  + 0.25,
        "ETA_INC":  BASE_ETA_INC  + 0.20,
        "HUB_ACTIVE_FRACTION": 1.0,
        "NEW_HUB_COUNT":       max(BASE_NEW_HUB_COUNT, 20),
        "NEW_HUB_NEAR_SECONDARY": True,
        "HUB_TRADES_AS_WEALTH_BONUS": BASE_HUB_TRADES_AS_WEALTH_BONUS * 2.2,
        "SEC_TRADE_MULT":      BASE_SEC_TRADE_MULT * 2.2,
        "GAMMA_NEED_SAVINGS":  BASE_GAMMA_NEED_SAVINGS * 1.4,
        "RISK_UNEMP_MULT": max(1.0, RISK_UNEMP_MULT * 0.8),
    },
}

# ---------------- Torus mesafe / CE-linear coupling / Gini ----------------
def _manh(a, b):
    """Toroidal Manhattan distance between two grid cells."""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return min(dx, GRID - dx) + min(dy, GRID - dy)

def _compute_distance_field(points):
    """Return a grid where each cell stores distance to the nearest point."""
    field = np.full((GRID, GRID), GRID * 2, dtype=float)
    if not points:
        return field
    xs = np.arange(GRID)
    ys = np.arange(GRID)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    for (px, py) in points:
        dx = np.abs(X - px)
        dy = np.abs(Y - py)
        dx = np.minimum(dx, GRID - dx)
        dy = np.minimum(dy, GRID - dy)
        d = dx + dy
        field = np.minimum(field, d)
    return field

def update_regen_from_ce():
    """Update resource regeneration from the local CE flow field."""
    global regen, ce_intensity, ce_flow, regen_base
    if np.any(ce_flow > 0):
        max_flow = float(ce_flow.max())
        I = ce_flow / max_flow if max_flow > 0 else np.zeros_like(ce_flow)
        ce_intensity[:] = CE_MEMORY * ce_intensity + (1.0 - CE_MEMORY) * I

        factor = np.ones_like(regen_base)
        pri_mask = (ptype == PRIMARY)
        sec_mask = (ptype == SECONDARY)

        LAMBDA_CE_PLUS = LAMBDA_PLUS
        LAMBDA_PRI_DEMAND = LAMBDA_MINUS

        factor[sec_mask] = 1.0 + LAMBDA_CE_PLUS * ce_intensity[sec_mask]
        factor[pri_mask] = 1.0 - LAMBDA_PRI_DEMAND * ce_intensity[pri_mask]
        factor = np.clip(factor, 0.7, 1.3)

        regen[:] = regen_base * factor
        ce_flow[:] = 0.0
    else:
        regen[:] = regen_base

def safe_gini(a):
    """Compute a Gini index for non-negative arrays, returning zero when undefined."""
    x = np.sort(np.asarray(a, float))
    if x.size <= 1:
        return 0.0
    S = x.sum()
    if S <= 0:
        return 0.0
    n = x.size
    return ((2 * np.arange(1, n+1) - n - 1).dot(x)) / (n * S)

# ---------- Advantage helpers ----------
def raw_cell_advantage(x, y):
    """Raw signed spatial-access score before normalization.

    Positive terms capture access to CE infrastructure; the negative term captures
    proximity to waste-burden locations. The raw value can be negative, so KPI K5
    uses the normalized non-negative field produced by compute_advantage_fields().
    """
    d_h  = dist_hub[x, y]       if dist_hub is not None       else GRID * 2
    d_w  = dist_waste[x, y]     if dist_waste is not None     else GRID * 2
    d_c  = dist_container[x, y] if dist_container is not None else GRID * 2
    d_rc = dist_recycling[x, y] if dist_recycling is not None else GRID * 2

    waste_term = 1.0 / (d_w + 1.0)

    if not CE_ACTIVE:
        hub_term = 0.0
        cont_term = 0.0
        recycl_term = 0.0
        near_sec = 0.0
    else:
        hub_term    = ADV_HUB_WEIGHT    / (d_h  + 1.0)
        cont_term   = ADV_CONT_WEIGHT   / (d_c  + 1.0)
        recycl_term = ADV_RECYCL_WEIGHT / (d_rc + 1.0)
        near_sec    = 1.0 if ptype[x, y] == SECONDARY else 0.0

    return (hub_term +
            cont_term +
            recycl_term +
            ADV_SEC_BONUS * near_sec -
            ADV_WASTE_PENALTY * waste_term)


def compute_advantage_fields():
    """Build raw and normalized spatial-advantage fields for the active scenario."""
    A_raw = np.zeros((GRID, GRID), dtype=float)
    for x in range(GRID):
        for y in range(GRID):
            A_raw[x, y] = raw_cell_advantage(x, y)

    finite = np.isfinite(A_raw)
    if not finite.any():
        A_plus = np.full_like(A_raw, 0.5, dtype=float)
        return A_raw, A_plus

    amin = float(np.nanmin(A_raw[finite]))
    amax = float(np.nanmax(A_raw[finite]))
    if abs(amax - amin) < 1e-12:
        A_plus = np.full_like(A_raw, 0.5, dtype=float)
    else:
        A_plus = (A_raw - amin) / (amax - amin)
        A_plus = np.clip(A_plus, 0.0, 1.0)
    return A_raw, A_plus


def household_advantage_raw(h):
    x, y = h.pos
    if advantage_raw_field is not None:
        return float(advantage_raw_field[x, y])
    return float(raw_cell_advantage(x, y))


def household_advantage_norm(h):
    x, y = h.pos
    if advantage_norm_field is not None:
        return float(advantage_norm_field[x, y])
    _, A_plus = compute_advantage_fields()
    return float(A_plus[x, y])


def household_advantage(h):
    return household_advantage_norm(h)

# ---------------- Scenario-specific micro-level policy effects ----------------
def apply_policy_micro_effects(h, municipality):
    """Apply scenario-level knowledge, opportunity and incentive effects to a household."""
    edu = float(municipality.eta_know)
    h.b_skill = float(np.clip(h.b_skill * (1.0 - 0.30 * edu), 0.0, 1.0))
    h.b_stig  = float(np.clip(h.b_stig  * (1.0 - 0.25 * edu), 0.0, 1.0))
    h.env     = float(np.clip(h.env + 0.10 * edu * (1.0 - h.env), 0.0, 1.0))

    opp = float(municipality.eta_opp)
    h.b_time = float(np.clip(h.b_time * (1.0 - 0.35 * opp), 0.0, 1.0))

    inc = float(municipality.eta_inc)
    h.b_cost = float(np.clip(h.b_cost * (1.0 - 0.30 * inc), 0.0, 1.0))

# ---------------- Agents ----------------
class Household:
    """Representative household decision unit.

    A household stores location, accumulated resources, CE motivations/barriers,
    recognition, secondary-material stock and bulky-waste stock. It is not a
    biological individual; household size is represented separately and affects
    product need and waste generation through a demographic-load factor.
    """
    _uid = 0

    def __init__(self, x, y, wealth, voice_base, recog, gen=0,
                 need=NEED, hub_trade_bonus=HUB_TRADES_AS_WEALTH_BONUS,
                 age=None, household_size=None):
        self.id = Household._uid
        Household._uid += 1

        self.pos = (x, y)
        self.wealth = float(wealth)
        self.voice  = float(voice_base)
        self.recognition = neutral_initial_recognition() if RESET_INITIAL_RECOGNITION else float(recog)
        self.sec   = 0.0
        self.bulky = 0.0
        self.gen   = gen
        self.age = float(age) if age is not None else float(sample_initial_age())

        if household_size is None:
            household_size = sample_household_size()
        self.household_size = int(np.clip(int(round(household_size)), HOUSEHOLD_SIZE_MIN, HOUSEHOLD_SIZE_MAX))

        self.need_base_unit = float(need)
        self.need_savings  = 0.0
        self.hub_trade_bonus = float(hub_trade_bonus)
        self._update_need_from_size()

        base = np.array([ENV_WEIGHT, SOCIAL_WEIGHT, FIN_WEIGHT], dtype=float)
        base = np.clip(base, 1e-3, None)
        base /= base.sum()
        alpha = base * float(MOTIVE_CONC)
        env_soc_fin = np.random.dirichlet(alpha)
        self.env = float(env_soc_fin[0])
        self.soc = float(env_soc_fin[1])
        self.fin = float(env_soc_fin[2])

        self.b_cost  = float(np.clip(np.random.beta(3, 2), 0, 1))
        self.b_time  = float(np.clip(np.random.beta(2, 3), 0, 1))
        self.b_skill = float(np.clip(np.random.beta(2, 2), 0, 1))
        self.b_stig  = float(np.clip(np.random.beta(2, 3), 0, 1))

        seg_p = np.asarray(SORT_SEGMENT_SHARES, dtype=float)
        if seg_p.sum() <= 0:
            seg_p = np.ones_like(seg_p) / len(seg_p)
        else:
            seg_p = seg_p / seg_p.sum()

        self.sort_segment = int(np.random.choice(len(seg_p), p=seg_p))
        self.sort_time_budget = SEG_SORT_TIME_MINUTES[self.sort_segment] / 60.0
        self.sort_value_of_time = float(np.random.uniform(SORT_TIME_VALUE_MIN, SORT_TIME_VALUE_MAX))
        self.sort_pref = float(np.clip(np.random.beta(3, 2), 0.05, 0.95))

        self.lin_exposure = float(np.clip(np.random.beta(2, 2), 0.0, 1.0))
        base_ce = 0.5 * self.env + 0.5 * (1.0 - self.b_skill)
        self.ce_employability = float(np.clip(base_ce + np.random.normal(0, 0.1), 0.0, 1.0))

        self.J_prim = float(np.clip(0.6 + 0.4 * self.lin_exposure + np.random.normal(0, 0.05), 0.2, 1.0))
        self.J_CE = float(np.clip(0.2 * self.ce_employability + np.random.normal(0, 0.03), 0.0, 0.8))

        risk_score = 0
        if self.lin_exposure > 0.6:
            risk_score += 1
        if self.ce_employability < 0.4:
            risk_score += 1
        if self.env < 0.4:
            risk_score += 1
        self.risk_group = bool(risk_score >= 2)

        self.employed = True
        self.last_income_primary = 0.0
        self.traded_this_cycle = False

    def _update_need_from_size(self):
        """Scale product need by household size using sub-linear shared consumption."""
        factor = household_size_factor(self.household_size)
        self.need = float(max(0.1, self.need_base_unit * factor))
        self.need_baseline = float(self.need)

    def update_household_size(self) -> bool:
        """Occasionally adjust household size without creating child-household agents."""
        if (not USE_HOUSEHOLD_SIZE_DYNAMICS) or random.random() >= HOUSEHOLD_SIZE_CHANGE_PROB:
            return False

        old_size = int(self.household_size)
        if old_size <= HOUSEHOLD_SIZE_MIN:
            new_size = old_size + 1
        elif old_size >= HOUSEHOLD_SIZE_MAX:
            new_size = old_size - 1
        else:
            new_size = old_size + (1 if random.random() < HOUSEHOLD_SIZE_UP_PROB else -1)

        new_size = int(np.clip(new_size, HOUSEHOLD_SIZE_MIN, HOUSEHOLD_SIZE_MAX))
        if new_size != old_size:
            self.household_size = new_size
            self._update_need_from_size()
            return True
        return False

    def _maybe_add_recognition(self, increment: float) -> bool:
        """Add recognition only when a CE action is socially observed."""
        global month_observed_ce_actions, month_recognition_opportunities
        month_recognition_opportunities += 1
        if random.random() < OBSERVABILITY_PROB:
            self.recognition = float(np.clip(self.recognition + increment, 0.0, 1.0))
            month_observed_ce_actions += 1
            return True
        return False

    def recognition_action_bonus(self) -> float:
        return float(W_RECOG_ACTION * self.soc * self.recognition)

    def local_recycling_capacity(self, radius: int = 1) -> float:
        x0, y0 = self.pos
        cnt = 0
        tot = 0
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx = (x0 + dx) % GRID
                ny = (y0 + dy) % GRID
                tot += 1
                if ptype[nx, ny] == SECONDARY:
                    cnt += 1
        if tot == 0:
            return 0.0
        return float(cnt) / float(tot)

    def harvest(self):
        if not self.employed:
            return
        x, y = self.pos
        c = ptype[x, y]
        if c != PRIMARY or stock[x, y] <= 0:
            return
        got = min(stock[x, y], self.need)
        stock[x, y] -= got
        self.last_income_primary += got

    def utility(self, xy):
        """Exposure-update utility for candidate grid cells.

        This is not residential migration; it is a stylized update of the
        household's effective spatial exposure under the scenario assumptions.
        """
        x, y = xy
        d_h = dist_hub[x, y]   if dist_hub is not None   else GRID * 2
        d_w = dist_waste[x, y] if dist_waste is not None else GRID * 2
        safe_wealth_log = math.log(max(1.0, self.wealth))

        sec_pull = 0.0
        if self.sec > 0:
            sec_pull = W_SEC_HUB * (0.5 + self.env) / (d_h + 1.0)

        U_env = (-W_DIST_HUB * d_h - W_WASTE_EXP / (d_w + 1.0) + sec_pull)
        U_soc = W_RECOG_BON * self.recognition
        U_fin = W_WEALTH_INERTIA * safe_wealth_log

        return self.env * U_env + self.soc * U_soc + self.fin * U_fin

    def generate_and_sort_waste(self, municipality, season_factor):
        """Generate household waste and probabilistically sort part of small waste.

        Sorted material enters the household secondary-material stock. Delivered
        flow is counted later when the household performs a container drop-off.
        """
        global ce_flow
        if not CE_ACTIVE:
            return

        need_eff = max(1.0, self.need_baseline * (1.0 - 0.35 * self.need_savings))
        extra = 0.15 * math.log1p(max(self.wealth, 0.0))
        waste_flow = max(0.0, need_eff + extra)
        if waste_flow <= 0:
            return

        bulky_share = float(np.clip(BULKY_SHARE, 0.0, 0.9))
        small_waste = waste_flow * (1.0 - bulky_share)
        bulky_waste = waste_flow * bulky_share

        if bulky_waste > 0:
            self.bulky += bulky_waste
        if small_waste <= 0:
            return

        seg = self.sort_segment
        cap_local = self.local_recycling_capacity(radius=1)
        norm_wealth = math.tanh(self.wealth / 50.0)

        M_i = (0.7 * self.env + 0.3 * self.soc) * (1.0 + municipality.eta_know)
        M_i += SEG_CAPACITY_WEIGHT[seg] * cap_local
        C_i = 0.5 * norm_wealth + 0.5 * (1.0 - self.b_skill)
        O_i = cap_local * (1.0 + municipality.eta_opp) * (1.0 - 0.5 * self.b_time)

        SORT_MIN_PER_UNIT = 3.0
        time_required_h = (SORT_MIN_PER_UNIT * small_waste) / 60.0
        time_ratio = time_required_h / max(1e-3, self.sort_time_budget)
        time_cost_eur = self.sort_value_of_time * time_required_h
        time_term = 0.5 * time_ratio + 0.5 * (time_cost_eur / 10.0)
        time_penalty = GAMMA_TIME_COST * SEG_TIME_SENSITIVITY[seg] * time_term
        social_bonus = self.recognition_action_bonus()

        z_sort = (ALPHA_MOT * M_i + ALPHA_CAP * C_i + ALPHA_OPP * O_i + social_bonus -
                  BETA_COST * self.b_cost - BETA_SKILL * self.b_skill -
                  BETA_STIG * self.b_stig - time_penalty)
        z_sort *= season_factor
        p_sort = float(np.clip(_stable_sigmoid(z_sort), 0.02, 0.98))

        if random.random() < p_sort:
            sort_share = self.sort_pref
        else:
            sort_share = 0.2 * self.sort_pref

        sort_share = float(np.clip(sort_share, 0.0, 1.0))
        sorted_mass = small_waste * sort_share
        if sorted_mass <= 0:
            return

        self.sec += sorted_mass
        x, y = self.pos
        ce_flow[x, y] += 0.3 * sorted_mass

    def ce_action(self, firms, season_trade_factor, municipality):
        """Evaluate monthly CE events: container drop-off, bulky drop-off and repair visit."""
        global month_trades_count, month_trades_mass, month_repairs_count, ce_flow
        global month_sort_dropoff_count, month_sort_dropoff_mass
        global month_bulky_dropoff_count, month_bulky_dropoff_mass

        if not CE_ACTIVE:
            return

        x, y = self.pos

        # 1) Sorting-container drop-off
        if self.sec > 0 and container_cells:
            d_c = dist_container[x, y] if dist_container is not None else GRID * 2
            opp_dist_c = 1.0 / (1.0 + max(0.0, d_c))
            seg = self.sort_segment
            cap_local = self.local_recycling_capacity(radius=1)
            norm_wealth = math.tanh(self.wealth / 50.0)

            M_i = (0.7 * self.env + 0.3 * self.soc) * (1.0 + municipality.eta_know)
            M_i += SEG_CAPACITY_WEIGHT[seg] * cap_local
            C_i = 0.5 * norm_wealth + 0.5 * (1.0 - self.b_skill)
            O_i = opp_dist_c * (1.0 + municipality.eta_opp) * (1.0 - 0.3 * self.b_time)

            time_required_h = (1.0 * d_c) / 60.0
            time_ratio = time_required_h / max(1e-3, self.sort_time_budget)
            time_cost_eur = self.sort_value_of_time * time_required_h
            time_term = 0.5 * time_ratio + 0.5 * (time_cost_eur / 10.0)
            time_penalty = GAMMA_TIME_COST * SEG_TIME_SENSITIVITY[seg] * time_term
            social_bonus = self.recognition_action_bonus()

            z_drop_c = (ALPHA_MOT * M_i + ALPHA_CAP * C_i + ALPHA_OPP * O_i + social_bonus -
                        BETA_COST * self.b_cost - BETA_TIME * self.b_time -
                        BETA_SKILL * self.b_skill - BETA_STIG * self.b_stig - time_penalty)
            z_drop_c *= season_trade_factor
            p_drop_c = float(np.clip(_stable_sigmoid(z_drop_c), 0.02, 0.98))

            if random.random() < p_drop_c:
                dropped_mass = self.sec
                self.sec = 0.0
                month_trades_count += 1
                month_trades_mass  += dropped_mass
                month_sort_dropoff_count += 1
                month_sort_dropoff_mass  += dropped_mass
                ce_flow[x, y] += dropped_mass * SEC_TRADE_MULT
                self._maybe_add_recognition(0.05)
                self.need_savings = float(np.clip(self.need_savings + 0.03 * self.hub_trade_bonus, 0.0, 1.0))
                self.traded_this_cycle = True

        # 1b) Bulky-waste drop-off
        if self.bulky > BULKY_THRESHOLD and recycling_cells:
            d_rc = dist_recycling[x, y] if dist_recycling is not None else GRID * 2
            opp_dist_rc = 1.0 / (1.0 + max(0.0, d_rc))
            seg = self.sort_segment
            norm_wealth = math.tanh(self.wealth / 50.0)

            M_bulk = (0.6 * self.env + 0.4 * self.fin) * (1.0 + municipality.eta_know)
            C_bulk = 0.4 * norm_wealth + 0.6 * (1.0 - self.b_skill)
            O_bulk = opp_dist_rc * (1.0 + municipality.eta_opp) * (1.0 - 0.4 * self.b_time)

            time_required_h_bulk = (3.0 * d_rc) / 60.0
            time_ratio_bulk = time_required_h_bulk / max(1e-3, self.sort_time_budget)
            time_cost_eur_bulk = self.sort_value_of_time * time_required_h_bulk
            time_term_bulk = 0.5 * time_ratio_bulk + 0.5 * (time_cost_eur_bulk / 10.0)
            time_penalty_bulk = GAMMA_TIME_COST * SEG_TIME_SENSITIVITY[seg] * time_term_bulk
            social_bonus = self.recognition_action_bonus()

            z_drop_bulk = (ALPHA_MOT * M_bulk + ALPHA_CAP * C_bulk + ALPHA_OPP * O_bulk + social_bonus -
                           BETA_COST * self.b_cost - BETA_TIME * self.b_time -
                           BETA_SKILL * self.b_skill - BETA_STIG * self.b_stig - time_penalty_bulk)
            z_drop_bulk *= season_trade_factor * (1.0 + 0.5 * municipality.eta_inc)
            p_drop_bulk = float(np.clip(_stable_sigmoid(z_drop_bulk), 0.01, 0.95))

            if random.random() < p_drop_bulk:
                dropped_bulk = self.bulky
                self.bulky = 0.0
                month_trades_count += 1
                month_trades_mass  += dropped_bulk
                month_bulky_dropoff_count += 1
                month_bulky_dropoff_mass  += dropped_bulk
                ce_flow[x, y] += dropped_bulk * SEC_TRADE_MULT * BULKY_CE_MULT
                self._maybe_add_recognition(0.10)
                self.need_savings = float(np.clip(self.need_savings + 0.12 * self.hub_trade_bonus, 0.0, 1.0))
                self.traded_this_cycle = True

        # 2) Repair-cafe visit
        if not firms:
            return
        hub_locs = list(firms.keys())
        if not hub_locs:
            return

        dists_h = [_manh(self.pos, loc) for loc in hub_locs]
        d_min_h = min(dists_h)
        opp_dist_h = 1.0 / (1.0 + max(0, d_min_h))
        seg = self.sort_segment
        norm_wealth = math.tanh(self.wealth / 50.0)

        M_rep = (0.8 * self.env + 0.2 * self.soc) * (1.0 + municipality.eta_know)
        C_rep = 0.3 * norm_wealth + 0.7 * (1.0 - self.b_skill)
        O_rep = opp_dist_h * (1.0 + municipality.eta_opp) * (1.0 - 0.3 * self.b_time)

        time_required_h = (2.0 * d_min_h) / 60.0
        time_ratio_rep = time_required_h / max(1e-3, self.sort_time_budget)
        time_cost_eur_rep = self.sort_value_of_time * time_required_h
        time_term_rep = 0.5 * time_ratio_rep + 0.5 * (time_cost_eur_rep / 10.0)
        time_penalty_rep = GAMMA_TIME_COST * SEG_TIME_SENSITIVITY[seg] * time_term_rep
        social_bonus = self.recognition_action_bonus()

        z_rep = (ALPHA_MOT * M_rep + ALPHA_CAP * C_rep + ALPHA_OPP * O_rep + social_bonus -
                 BETA_COST * self.b_cost - BETA_TIME * self.b_time -
                 BETA_SKILL * self.b_skill - BETA_STIG * self.b_stig - time_penalty_rep)
        z_rep *= season_trade_factor
        p_rep = float(np.clip(_stable_sigmoid(z_rep), 0.02, 0.98))

        if random.random() > p_rep:
            return

        idx = int(np.argmin(dists_h))
        loc = hub_locs[idx]
        f = firms[loc]
        f.repair_visits += 1
        month_repairs_count += 1
        ce_flow[x, y] += self.need_baseline * 0.5 * SEC_TRADE_MULT
        self._maybe_add_recognition(0.10)
        self.need_savings = float(np.clip(self.need_savings + 0.08 * self.hub_trade_bonus, 0.0, 1.0))
        self.traded_this_cycle = True

    def move(self, occ):
        x0, y0 = self.pos
        best_u = self.utility((x0, y0))
        candidates = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx, ny = (x0 + dx) % GRID, (y0 + dy) % GRID
                if (nx, ny) == (x0, y0):
                    continue
                if occ[nx, ny] >= MAX_PATCH_CAPACITY:
                    continue
                u = self.utility((nx, ny))
                candidates.append(((nx, ny), u + np.random.normal(0, 0.08)))

        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            (nx, ny), max_u = candidates[0]
            if max_u > best_u:
                occ[x0, y0] = max(0, occ[x0, y0] - 1)
                self.pos = (nx, ny)
                occ[nx, ny] += 1
                return True
        return False

    def socialise(self, occmap):
        nbrs = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = (self.pos[0] + dx) % GRID, (self.pos[1] + dy) % GRID
                n = occmap.get((nx, ny))
                if n and n is not self:
                    nbrs.append(n)

        if nbrs:
            max_rec_nbr = max(n.recognition for n in nbrs)
            delta = BETA_RECOG * (max_rec_nbr - self.recognition)
            self.recognition = float(np.clip(self.recognition + delta + np.random.normal(0, 0.025), 0.0, 1.0))

    def reproduce(self, agents, occ):
        global repro_attempts, repro_success
        age_here = getattr(self, "age", 30.0)
        if (self.wealth > REPRO_THRESHOLD and
            REPRO_MIN_AGE <= age_here <= REPRO_MAX_AGE and
            random.random() < REPRO_PROB):
            repro_attempts += 1
            local = [((self.pos[0] + dx) % GRID, (self.pos[1] + dy) % GRID)
                     for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
            empt_local = [c for c in local if occ[c] == 0 and ptype[c] != WASTE]
            targets = empt_local if empt_local else [
                (i, j) for i in range(GRID) for j in range(GRID)
                if occ[i, j] == 0 and ptype[i, j] != WASTE
            ]
            if targets:
                spot = random.choice(targets)
                child = Household(*spot,
                                  wealth=self.wealth * REPRO_SHARE,
                                  voice_base=self.voice,
                                  recog=self.recognition,
                                  gen=self.gen + 1,
                                  need=self.need,
                                  hub_trade_bonus=self.hub_trade_bonus,
                                  age=0.0)
                self.wealth *= (1 - REPRO_SHARE)
                agents.append(child)
                occ[spot] += 1
                repro_success += 1

    def step(self, occ, firms, occmap, season_trade_factor, municipality):
        apply_policy_micro_effects(self, municipality)
        moved = False
        if random.random() < MOVE_PROB:
            moved = self.move(occ)
        self.harvest()
        self.generate_and_sort_waste(municipality, season_trade_factor)
        self.ce_action(firms, season_trade_factor, municipality)
        self.socialise(occmap)
        return moved

class Firm:
    def __init__(self, loc):
        self.loc = loc
        self.repair_visits = 0
        self.total_repairs = 0

    def step(self):
        self.total_repairs += self.repair_visits
        self.repair_visits = 0

class Municipality:
    def __init__(self, eta_know, eta_opp, eta_inc):
        self.eta_know = float(eta_know)
        self.eta_opp  = float(eta_opp)
        self.eta_inc  = float(eta_inc)

# ---------------- Plot helpers ----------------
def plot_infrastructure(out_pdf_name):
    fig, ax = plt.subplots(figsize=(11.5, 10.5))
    fig.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.14)
    base = np.ones_like(ptype, float)
    ax.imshow(base.T, origin="lower", cmap="Greys", vmin=0, vmax=1, alpha=0.08, zorder=0)

    if container_cells:
        cx, cy = zip(*container_cells)
        ax.scatter(cx, cy, marker="^", s=55, facecolors="none", edgecolors="black", linewidths=1.0, zorder=3, label="Sorting container")
    if recycling_cells:
        rx, ry = zip(*recycling_cells)
        ax.scatter(rx, ry, marker="D", s=55, facecolors="none", edgecolors="black", linewidths=1.0, zorder=3, label="Recycling centre")
    if hub_cells:
        hx, hy = zip(*hub_cells)
        ax.scatter(hx, hy, marker="*", s=95, facecolors="none", edgecolors="black", linewidths=1.0, zorder=4, label="Repair café (hub)")
    if waste_cells:
        wx, wy = zip(*waste_cells)
        ax.scatter(wx, wy, marker="X", s=95, c="black", linewidths=1.0, zorder=4, label="Waste hotspot")

    handles = [
        Line2D([0], [0], marker="^", linestyle="", markerfacecolor="none", markeredgecolor="black", markersize=8, label="Sorting container"),
        Line2D([0], [0], marker="D", linestyle="", markerfacecolor="none", markeredgecolor="black", markersize=8, label="Recycling centre"),
        Line2D([0], [0], marker="*", linestyle="", markerfacecolor="none", markeredgecolor="black", markersize=10, label="Repair café (hub)"),
        Line2D([0], [0], marker="X", linestyle="", markerfacecolor="black", markeredgecolor="black", markersize=8, label="Waste hotspot"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False, columnspacing=1.8, handletextpad=0.7, fontsize=15)
    ax.set_title("Amsterdam Grid: CE Infrastructure\nCycle 0", fontsize=22, pad=10)
    ax.set_xlim(-0.5, GRID - 0.5)
    ax.set_ylim(-0.5, GRID - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(RUN_DIR / out_pdf_name, format="pdf", bbox_inches="tight")
    plt.close(fig)

def _per_household_value(h, field_name):
    if field_name == "env":
        return h.env
    if field_name == "recognition":
        return h.recognition
    if field_name == "advantage":
        return household_advantage_norm(h)
    if field_name == "advantage_raw":
        return household_advantage_raw(h)
    if field_name == "wealth":
        return h.wealth
    if field_name == "resources":
        return h.wealth
    return h.wealth

def _cell_mean_matrix(households, field_name):
    vals_sum = np.zeros((GRID, GRID), dtype=float)
    counts   = np.zeros((GRID, GRID), dtype=int)
    for h in households:
        v = _per_household_value(h, field_name)
        x, y = h.pos
        vals_sum[x, y] += v
        counts[x, y]   += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_mat = vals_sum / counts
    mean_mat[counts == 0] = np.nan
    return mean_mat

def plot_cell_mean(cycle, households, field_name, title_prefix):
    if not households:
        return
    cell_mean = _cell_mean_matrix(households, field_name)
    valid = np.isfinite(cell_mean)
    if not np.any(valid):
        return
    vmin, vmax = np.nanpercentile(cell_mean[valid], [5, 95])
    if vmin == vmax:
        vmin -= 1e-6
        vmax += 1e-6

    fig, ax = plt.subplots(figsize=(13.0, 11.0))
    fig.subplots_adjust(left=0.06, right=0.86, top=0.89, bottom=0.13)
    ax.set_facecolor("#e0e0e0")
    im = ax.imshow(cell_mean.T, origin="lower", cmap="plasma", vmin=vmin, vmax=vmax, alpha=0.95, zorder=1)

    px, py = np.where(ptype == PRIMARY)
    if px.size > 0:
        ax.scatter(px, py, marker="s", s=28, facecolors="none", edgecolors="black", linewidths=0.55, zorder=2.0)
    if container_cells:
        cx, cy = zip(*container_cells)
        ax.scatter(cx, cy, marker="^", s=70, facecolors="none", edgecolors="black", linewidths=1.0, zorder=3.0)
    if recycling_cells:
        rx, ry = zip(*recycling_cells)
        ax.scatter(rx, ry, marker="D", s=75, facecolors="none", edgecolors="black", linewidths=1.0, zorder=3.1)
    if hub_cells:
        hx, hy = zip(*hub_cells)
        ax.scatter(hx, hy, marker="*", s=220, facecolors="white", edgecolors="#00bcd4", linewidths=1.4, zorder=3.2)
    if waste_cells:
        wx, wy = zip(*waste_cells)
        ax.scatter(wx, wy, marker="X", s=130, facecolors="black", edgecolors="black", linewidths=1.2, zorder=3.3)

    if field_name == "advantage":
        c_label = "Normalized spatial-advantage exposure (cell mean)"
    elif field_name == "advantage_raw":
        c_label = "Raw spatial-access score (cell mean)"
    elif field_name in ("wealth", "resources"):
        c_label = "Accumulated economic resources (cell mean)"
    else:
        c_label = field_name

    cbar = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.01, shrink=0.85, aspect=25)
    cbar.set_label(c_label, fontsize=20)
    cbar.ax.tick_params(labelsize=16)
    ax.set_title(f"{title_prefix}\nCycle {cycle} (Month)", fontsize=22, pad=10)
    ax.set_xlim(-0.5, GRID - 0.5)
    ax.set_ylim(-0.5, GRID - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])

    handles = [
        Line2D([0], [0], marker="s", linestyle="", markerfacecolor="none", markeredgecolor="black", markersize=8, label="Primary income patch"),
        Line2D([0], [0], marker="^", linestyle="", markerfacecolor="none", markeredgecolor="black", markersize=8, label="Sorting container"),
        Line2D([0], [0], marker="D", linestyle="", markerfacecolor="none", markeredgecolor="black", markersize=8, label="Recycling centre"),
        Line2D([0], [0], marker="*", linestyle="", markerfacecolor="white", markeredgecolor="#00bcd4", markersize=10, label="Repair café (hub)"),
        Line2D([0], [0], marker="X", linestyle="", markerfacecolor="black", markeredgecolor="black", markersize=8, label="Waste hotspot"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.045), ncol=3, frameon=False, columnspacing=1.6, handletextpad=0.7, fontsize=15)
    out_name = RUN_DIR / f"grid_cell_{cycle:03d}_{field_name}.pdf"
    fig.savefig(out_name, format="pdf", bbox_inches="tight")
    plt.close(fig)

# ---------------- Run one policy scenario simulation ----------------
def run_sim(policy_name, policy_cfg):
    """Run one scenario/replicate and write monthly metrics to the run directory."""
    global RUN_DIR, ptype, stock, regen_base, regen, ce_intensity, ce_flow
    global hub_cells, waste_cells, container_cells, recycling_cells, hub_fraction
    global advantage_raw_field, advantage_norm_field
    global GRID, N_HOUSEHOLDS_SCALE, HUB_ACTIVE_FRACTION, NEW_HUB_COUNT, NEW_HUB_NEAR_SECONDARY
    global ETA_KNOW, ETA_OPP, ETA_INC, LAMBDA_PLUS, LAMBDA_MINUS
    global UNEMP_BETA_CEI, RISK_UNEMP_MULT, MOVE_PROB, OBSERVABILITY_PROB, W_RECOG_ACTION
    global SEC_TRADE_MULT, HUB_TRADES_AS_WEALTH_BONUS
    global month_trades_count, month_trades_mass, month_repairs_count
    global month_sort_dropoff_count, month_sort_dropoff_mass
    global month_bulky_dropoff_count, month_bulky_dropoff_mass
    global month_move_count, month_size_change_count, month_observed_ce_actions, month_recognition_opportunities
    global repro_attempts, repro_success
    global dist_hub, dist_waste, dist_container, dist_recycling
    global curr_unemployment, CE_ACTIVE, GAMMA_NEED_SAVINGS

    print(f"\n=== Policy scenario: {policy_name} ===")

    run_seed = int(policy_cfg.get("SEED", SIM_SEED))
    random.seed(run_seed)
    np.random.seed(run_seed)

    run_tag = policy_cfg.get("_RUN_TAG") or policy_cfg.get("RUN_TAG") or policy_cfg.get("REPLICATE")
    if run_tag is None:
        RUN_DIR = RUNS_DIR / f"{BASE_RUN_NAME}_{policy_name}"
    else:
        safe_tag = str(run_tag).replace(" ", "_")
        RUN_DIR = RUNS_DIR / f"{BASE_RUN_NAME}_{safe_tag}_{policy_name}"
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    # Compact K1 spatial diagnostics are saved as arrays, not as PDFs. They are
    # intentionally separate from SAVE_SPATIAL_FIGURES, which controls heavy
    # per-cycle map exports.
    save_k1_spatial_data = bool(policy_cfg.get("SAVE_K1_SPATIAL_DATA", BASE_SAVE_K1_SPATIAL_DATA))
    k1_spatial_last_window = int(policy_cfg.get("K1_SPATIAL_LAST_WINDOW", BASE_K1_SPATIAL_LAST_WINDOW))
    k1_spatial_last_window = max(1, k1_spatial_last_window)

    CE_ACTIVE = policy_cfg.get("CE_ACTIVE", True)
    GAMMA_NEED_SAVINGS = policy_cfg.get("GAMMA_NEED_SAVINGS", BASE_GAMMA_NEED_SAVINGS)
    MOVE_PROB = float(np.clip(policy_cfg.get("MOVE_PROB", BASE_MOVE_PROB), 0.0, 1.0))
    OBSERVABILITY_PROB = float(np.clip(policy_cfg.get("OBSERVABILITY_PROB", BASE_OBSERVABILITY_PROB), 0.0, 1.0))
    W_RECOG_ACTION = float(max(0.0, policy_cfg.get("W_RECOG_ACTION", BASE_W_RECOG_ACTION)))
    UNEMP_BETA_CEI = float(policy_cfg.get("UNEMP_BETA_CEI", BASE_UNEMP_BETA_CEI))
    RISK_UNEMP_MULT = float(policy_cfg.get("RISK_UNEMP_MULT", BASE_RISK_UNEMP_MULT))

    N_HOUSEHOLDS_SCALE = policy_cfg.get("N_HOUSEHOLDS_SCALE", BASE_N_HOUSEHOLDS_SCALE)
    HUB_ACTIVE_FRACTION = policy_cfg.get("HUB_ACTIVE_FRACTION", BASE_HUB_ACTIVE_FRACTION)
    HUB_ACTIVE_FRACTION = max(0.0, min(HUB_ACTIVE_FRACTION, 1.0))
    NEW_HUB_COUNT = policy_cfg.get("NEW_HUB_COUNT", BASE_NEW_HUB_COUNT)
    NEW_HUB_NEAR_SECONDARY = policy_cfg.get("NEW_HUB_NEAR_SECONDARY", BASE_NEW_HUB_NEAR_SECONDARY)

    ETA_KNOW = policy_cfg.get("ETA_KNOW", BASE_ETA_KNOW)
    ETA_OPP  = policy_cfg.get("ETA_OPP",  BASE_ETA_OPP)
    ETA_INC  = policy_cfg.get("ETA_INC",  BASE_ETA_INC)

    LAMBDA_PLUS  = policy_cfg.get("LAMBDA_PLUS",  BASE_LAMBDA_PLUS)
    LAMBDA_MINUS = policy_cfg.get("LAMBDA_MINUS", BASE_LAMBDA_MINUS)

    SEC_TRADE_MULT = policy_cfg.get("SEC_TRADE_MULT", BASE_SEC_TRADE_MULT)
    HUB_TRADES_AS_WEALTH_BONUS = policy_cfg.get("HUB_TRADES_AS_WEALTH_BONUS", BASE_HUB_TRADES_AS_WEALTH_BONUS)

    ptype      = ptype_base.copy()
    stock      = stock_base.copy()
    regen_base = regen_base_base.copy()
    GRID       = ptype.shape[0]

    regen        = regen_base.copy()
    ce_intensity = np.zeros((GRID, GRID), dtype=float)
    ce_flow      = np.zeros((GRID, GRID), dtype=float)

    # K1 spatial diagnostics. Initial grid is zero by construction. We keep
    # compact running summaries and the last-window stack for heat-map figures.
    k1_initial_grid = ce_intensity.copy()
    k1_all_sum = np.zeros_like(ce_intensity, dtype=float)
    k1_all_sumsq = np.zeros_like(ce_intensity, dtype=float)
    k1_all_count = 0
    k1_last_window_grids = []
    k1_final_grid = ce_intensity.copy()

    hub_cells       = list(hub_cells_base)
    waste_cells     = list(waste_cells_base)
    container_cells = list(container_cells_base)
    recycling_cells = list(recycling_cells_base)
    hub_fraction    = dict(hub_fraction_base)

    rng_hub = np.random.default_rng(SIM_SEED + 123)
    if 0.0 < HUB_ACTIVE_FRACTION < 1.0 and len(hub_cells) > 0:
        n_total = len(hub_cells)
        n_keep  = max(1, int(round(n_total * HUB_ACTIVE_FRACTION)))
        chosen_idx = rng_hub.choice(np.arange(n_total), size=n_keep, replace=False)
        chosen_idx = set(int(i) for i in chosen_idx)
        active_hubs = []
        inactive_hubs = []
        for idx, cell in enumerate(hub_cells):
            if idx in chosen_idx:
                active_hubs.append(cell)
            else:
                inactive_hubs.append(cell)
        for (x, y) in inactive_hubs:
            if ptype[x, y] == HUB:
                ptype[x, y] = PRIMARY
        hub_cells = active_hubs
        hub_fraction = {loc: hub_fraction.get(loc, "RepairCafe") for loc in hub_cells}

    if NEW_HUB_COUNT > 0:
        candidates = []
        if NEW_HUB_NEAR_SECONDARY:
            for x in range(GRID):
                for y in range(GRID):
                    if ptype[x, y] == WASTE or (x, y) in hub_cells:
                        continue
                    near_sec = False
                    if ptype[x, y] == SECONDARY:
                        near_sec = True
                    else:
                        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                            xx = (x+dx) % GRID
                            yy = (y+dy) % GRID
                            if ptype[xx, yy] == SECONDARY:
                                near_sec = True
                                break
                    if near_sec:
                        candidates.append((x, y))
        else:
            candidates = [(x, y) for x in range(GRID) for y in range(GRID)
                          if ptype[x, y] != WASTE and (x, y) not in hub_cells]

        rng_new = np.random.default_rng(SIM_SEED + 789)
        if candidates:
            n_add = min(int(NEW_HUB_COUNT), len(candidates))
            chosen = rng_new.choice(len(candidates), size=n_add, replace=False)
            for idx in chosen:
                loc = candidates[int(idx)]
                hub_cells.append(loc)
                ptype[loc] = HUB
                hub_fraction[loc] = "RepairCafe"

    dist_hub       = _compute_distance_field(hub_cells)
    dist_waste     = _compute_distance_field(waste_cells)
    dist_container = _compute_distance_field(container_cells)
    dist_recycling = _compute_distance_field(recycling_cells)
    advantage_raw_field, advantage_norm_field = compute_advantage_fields()

    plot_infrastructure("grid_infrastructure_000.pdf")

    agents_raw = pickle.load(open(AGNT_FILE, "rb"))
    agents_raw = list(agents_raw)
    rng_hh = np.random.default_rng(SIM_SEED + 456)
    n0 = len(agents_raw)
    if N_HOUSEHOLDS_SCALE <= 0:
        N_HOUSEHOLDS_SCALE = 0.01
    n_target = max(1, int(round(n0 * N_HOUSEHOLDS_SCALE)))

    if n_target < n0:
        indices = rng_hh.choice(np.arange(n0), size=n_target, replace=False)
        agents_raw = [agents_raw[int(i)] for i in indices]
    elif n_target > n0:
        extra_idx = rng_hh.choice(np.arange(n0), size=(n_target - n0), replace=True)
        agents_raw = agents_raw + [agents_raw[int(i)] for i in extra_idx]

    Household._uid = 0

    def _household_from_record(rec):
        if len(rec) >= 7:
            x, y, w, v, r, gen, hh_size = rec[:7]
        else:
            x, y, w, v, r, gen = rec[:6]
            hh_size = None
        return Household(x, y, w, v, r, gen,
                         need=NEED,
                         hub_trade_bonus=HUB_TRADES_AS_WEALTH_BONUS,
                         household_size=hh_size)

    households = [_household_from_record(rec) for rec in agents_raw]
    occ_count = np.zeros((GRID, GRID), int)
    for h in households:
        occ_count[h.pos] += 1

    firms = {loc: Firm(loc) for loc in hub_cells}
    municipality = Municipality(ETA_KNOW, ETA_OPP, ETA_INC)

    repro_attempts = 0
    repro_success  = 0
    log = defaultdict(list)

    month_trades_count  = 0
    month_trades_mass   = 0.0
    month_repairs_count = 0
    month_sort_dropoff_count = 0
    month_sort_dropoff_mass  = 0.0
    month_bulky_dropoff_count = 0
    month_bulky_dropoff_mass  = 0.0
    month_move_count = 0
    month_size_change_count = 0
    month_observed_ce_actions = 0
    month_recognition_opportunities = 0

    curr_unemployment = UNEMP_REF

    plot_cell_mean(0, households, field_name="advantage", title_prefix="Amsterdam Grid - cell mean advantage")

    for cycle in range(1, CYCLES + 1):
        month_trades_count  = 0
        month_trades_mass   = 0.0
        month_repairs_count = 0
        month_sort_dropoff_count = 0
        month_sort_dropoff_mass  = 0.0
        month_bulky_dropoff_count = 0
        month_bulky_dropoff_mass  = 0.0
        month_move_count = 0
        month_size_change_count = 0
        month_observed_ce_actions = 0
        month_recognition_opportunities = 0

        if households:
            if USE_BIOLOGICAL_DEMOGRAPHY:
                for h in households:
                    if not hasattr(h, "age"):
                        h.age = float(sample_initial_age())
                    h.age += 1.0 / 12.0
                survivors = []
                for h in households:
                    p_death = monthly_death_prob(h.age)
                    if random.random() < p_death:
                        x, y = h.pos
                        occ_count[x, y] = max(0, occ_count[x, y] - 1)
                    else:
                        survivors.append(h)
                households = survivors
            else:
                for h in households:
                    if h.update_household_size():
                        month_size_change_count += 1

        if households:
            base_u = curr_unemployment
            for h in households:
                u_h = base_u
                if getattr(h, "risk_group", False):
                    u_h = min(1.0, base_u * RISK_UNEMP_MULT)
                p_emp = max(0.0, 1.0 - u_h)
                h.employed = (random.random() < p_emp)
                h.last_income_primary = 0.0
                h.traded_this_cycle = False
        else:
            curr_unemployment = 0.0

        update_regen_from_ce()

        m_idx = (cycle - 1) % 12
        season_regen = float(SEASON_REGEN[m_idx])
        season_trade = float(SEASON_TRADE[m_idx])
        stock[:] = np.minimum(stock + regen * season_regen, 140)

        occmap = {h.pos: h for h in households}
        random.shuffle(households)

        for h in households:
            if h.step(occ_count, firms, occmap, season_trade, municipality):
                month_move_count += 1

        for f in firms.values():
            f.step()

        for h in households:
            income_prim = max(0.0, float(h.last_income_primary))
            income_ce   = 0.0
            total_income = income_prim + income_ce
            base_need = max(0.0, float(getattr(h, "need_baseline", getattr(h, "need", NEED))))
            eff_need = base_need * (1.0 - GAMMA_NEED_SAVINGS * float(h.need_savings))
            eff_need = max(0.0, eff_need)
            delta_R = total_income - eff_need
            h.wealth += delta_R
            h.wealth = float(np.clip(h.wealth, EXCLUSION_THRESHOLD, None))

        if USE_BIOLOGICAL_DEMOGRAPHY and cycle % GEN_STEP == 0 and households:
            for h in list(households):
                h.reproduce(households, occ_count)

        if cycle % 12 == 0:
            plot_cell_mean(cycle, households, field_name="wealth", title_prefix="Amsterdam Grid - cell mean wealth")

        # ---- Metrics ----
        adv_raw_vals = np.array([household_advantage_raw(h) for h in households]) if households else np.array([])
        adv_vals     = np.array([household_advantage_norm(h) for h in households]) if households else np.array([])
        resources    = np.array([h.wealth for h in households], dtype=float) if households else np.array([])
        labor_income_vals = np.array([h.last_income_primary for h in households], dtype=float) if households else np.array([])

        D_labor_income = safe_gini(labor_income_vals) if labor_income_vals.size > 0 else 0.0
        D_adv_exposure = safe_gini(adv_vals)          if adv_vals.size          > 0 else 0.0
        D_resources    = safe_gini(resources)         if resources.size         > 0 else 0.0
        D_adv_cell     = safe_gini(advantage_norm_field.flatten()) if advantage_norm_field is not None else 0.0
        MEAN_ADV_EXPOSURE = float(np.nanmean(adv_vals)) if adv_vals.size > 0 else 0.0
        MEAN_RESOURCES    = float(np.nanmean(resources)) if resources.size > 0 else 0.0
        MEAN_LABOR_INCOME = float(np.nanmean(labor_income_vals)) if labor_income_vals.size > 0 else 0.0
        hh_sizes = np.array([getattr(h, "household_size", 1) for h in households], dtype=float) if households else np.array([])
        MEAN_HOUSEHOLD_SIZE = float(np.nanmean(hh_sizes)) if hh_sizes.size > 0 else 0.0
        TOTAL_SYNTHETIC_POPULATION = float(np.nansum(hh_sizes)) if hh_sizes.size > 0 else 0.0
        MOVE_RATE_M = month_move_count / max(1, len(households))
        OBSERVED_CE_ACTION_RATE_M = (
            month_observed_ce_actions / month_recognition_opportunities
            if month_recognition_opportunities > 0 else 0.0
        )

        D_adv    = D_adv_exposure
        D_wealth = D_resources

        TRADES_month_kg = month_trades_mass * UNIT_TO_KG
        SORT_DROPOFF_MASS_KG  = month_sort_dropoff_mass * UNIT_TO_KG
        BULKY_DROPOFF_MASS_KG = month_bulky_dropoff_mass * UNIT_TO_KG

        # K1: mean local CE activation.
        # ce_intensity is a grid-cell field updated from local CE flows with
        # local normalization and temporal memory; CE_INT is its spatial mean.
        CE_INT = float(ce_intensity.mean()) if ce_intensity.size > 0 else 0.0

        if save_k1_spatial_data and ce_intensity.size > 0:
            k1_grid_now = ce_intensity.copy()
            k1_all_sum += k1_grid_now
            k1_all_sumsq += k1_grid_now ** 2
            k1_all_count += 1
            k1_last_window_grids.append(k1_grid_now)
            if len(k1_last_window_grids) > k1_spatial_last_window:
                k1_last_window_grids.pop(0)
            k1_final_grid = k1_grid_now

        H = max(1, len(households))

        # K2: flow-based citywide CE index.
        # Material flow includes realized sorting-container and bulky-waste
        # drop-offs. Repair activity is measured as repair visits per household.
        # CE_INTENSITY is not included to avoid double-counting local activation.
        material_flow_norm = TRADES_month_kg / (H * NEED + 1e-9)
        repair_visit_norm  = month_repairs_count / H
        flow_weight_sum = max(1e-9, CEI_W_TRADE + CEI_W_REPAIR)
        w_material = CEI_W_TRADE / flow_weight_sum
        w_repair   = CEI_W_REPAIR / flow_weight_sum
        cei_sim_raw = w_material * material_flow_norm + w_repair * repair_visit_norm
        CEI_SIM = float(np.clip(cei_sim_raw, 0.0, 10.0))

        u_reg = UNEMP_REF + UNEMP_BETA_CEI * (CEI_SIM - CEI_REF)
        u_reg = float(np.clip(u_reg, 0.0, UNEMP_MAX))

        if households:
            J_tot = np.array([1.0 if h.employed else 0.0 for h in households], dtype=float)
            employ_mean = float(np.clip(J_tot.mean(), 0.0, 1.0))
            U_emp = 1.0 - employ_mean
            D_employ = safe_gini(J_tot)
            risk_mask = np.array([getattr(h, "risk_group", False) for h in households], dtype=bool)
            if risk_mask.any():
                J_tot_risk = J_tot[risk_mask]
                u_risk_emp = 1.0 - float(np.clip(J_tot_risk.mean(), 0.0, 1.0))
                R_gap = max(0.0, u_risk_emp - U_emp)
            else:
                R_gap = 0.0

            voice_arr = np.array([h.voice for h in households], dtype=float)
            adv_arr = adv_vals if adv_vals.size > 0 else np.zeros_like(voice_arr)
            if (voice_arr.size > 1 and np.std(voice_arr) > 1e-8 and np.std(adv_arr) > 1e-8):
                P_voice_corr = float(np.corrcoef(voice_arr, adv_arr)[0, 1])
            else:
                P_voice_corr = 0.0
        else:
            U_emp = 0.0
            D_employ = 0.0
            D_labor_income = 0.0
            R_gap = 0.0
            P_voice_corr = 0.0

        curr_unemployment = u_reg

        for k, v in [
            ("cycle", cycle),
            ("D_labor_income", D_labor_income),
            ("D_adv_exposure", D_adv_exposure),
            ("D_adv_cell", D_adv_cell),
            ("D_resources", D_resources),
            ("MEAN_ADV_EXPOSURE", MEAN_ADV_EXPOSURE),
            ("MEAN_RESOURCES", MEAN_RESOURCES),
            ("MEAN_LABOR_INCOME", MEAN_LABOR_INCOME),
            ("MEAN_HOUSEHOLD_SIZE", MEAN_HOUSEHOLD_SIZE),
            ("TOTAL_SYNTHETIC_POPULATION", TOTAL_SYNTHETIC_POPULATION),
            ("D_adv", D_adv),
            ("D_wealth", D_wealth),
            ("D_employ", D_employ),
            ("TRADES_COUNT_M", month_trades_count),
            ("TRADES_MASS_M", month_trades_mass),
            ("TRADES", TRADES_month_kg),
            ("SORT_DROPOFF_COUNT_M", month_sort_dropoff_count),
            ("SORT_DROPOFF_MASS_M", month_sort_dropoff_mass),
            ("SORT_DROPOFF_MASS_KG", SORT_DROPOFF_MASS_KG),
            ("BULKY_DROPOFF_COUNT_M", month_bulky_dropoff_count),
            ("BULKY_DROPOFF_MASS_M", month_bulky_dropoff_mass),
            ("BULKY_DROPOFF_MASS_KG", BULKY_DROPOFF_MASS_KG),
            ("REPAIRS_COUNT_M", month_repairs_count),
            ("MOVE_COUNT_M", month_move_count),
            ("MOVE_RATE_M", MOVE_RATE_M),
            ("HOUSEHOLD_SIZE_CHANGE_COUNT_M", month_size_change_count),
            ("OBSERVED_CE_ACTIONS_M", month_observed_ce_actions),
            ("CE_ACTION_RECOGNITION_OPPORTUNITIES_M", month_recognition_opportunities),
            ("OBSERVED_CE_ACTION_RATE_M", OBSERVED_CE_ACTION_RATE_M),
            ("N_HOUSEHOLDS", len(households)),
            ("MEAN_LOCAL_CE_ACTIVATION", CE_INT),
            ("CE_INTENSITY", CE_INT),
            ("CE_MATERIAL_FLOW_NORM", material_flow_norm),
            ("CE_REPAIR_VISIT_NORM", repair_visit_norm),
            ("CEI_SIM", CEI_SIM),
            ("UNEMPLOYMENT", u_reg),
            ("R_gap", R_gap),
            ("P_voice_corr", P_voice_corr),
        ]:
            log[k].append(v)

    k1_spatial_npz = None
    if save_k1_spatial_data:
        if k1_all_count > 0:
            k1_all_mean = k1_all_sum / float(k1_all_count)
            k1_all_var = np.maximum(k1_all_sumsq / float(k1_all_count) - k1_all_mean ** 2, 0.0)
            k1_all_sd = np.sqrt(k1_all_var)
        else:
            k1_all_mean = np.full_like(k1_initial_grid, np.nan, dtype=float)
            k1_all_sd = np.full_like(k1_initial_grid, np.nan, dtype=float)

        if k1_last_window_grids:
            k1_last_stack = np.stack(k1_last_window_grids, axis=0)
            k1_last_mean = np.nanmean(k1_last_stack, axis=0)
            k1_last_temporal_sd = np.nanstd(k1_last_stack, axis=0, ddof=1) if k1_last_stack.shape[0] > 1 else np.zeros_like(k1_last_mean)
        else:
            k1_last_mean = np.full_like(k1_initial_grid, np.nan, dtype=float)
            k1_last_temporal_sd = np.full_like(k1_initial_grid, np.nan, dtype=float)

        k1_change = k1_final_grid - k1_initial_grid
        k1_spatial_npz = RUN_DIR / "K1_spatial_ce_intensity.npz"
        np.savez_compressed(
            k1_spatial_npz,
            scenario=str(policy_name),
            seed=int(run_seed),
            cycles=int(CYCLES),
            grid_size=int(GRID),
            last_window=int(k1_spatial_last_window),
            initial_grid=k1_initial_grid,
            final_grid=k1_final_grid,
            change_grid=k1_change,
            all_months_mean_grid=k1_all_mean,
            all_months_sd_grid=k1_all_sd,
            last_window_mean_grid=k1_last_mean,
            last_window_temporal_sd_grid=k1_last_temporal_sd,
            ptype=ptype.astype(int),
            hub_cells=np.asarray(hub_cells, dtype=int) if hub_cells else np.empty((0, 2), dtype=int),
            waste_cells=np.asarray(waste_cells, dtype=int) if waste_cells else np.empty((0, 2), dtype=int),
            container_cells=np.asarray(container_cells, dtype=int) if container_cells else np.empty((0, 2), dtype=int),
            recycling_cells=np.asarray(recycling_cells, dtype=int) if recycling_cells else np.empty((0, 2), dtype=int),
        )

    print("Simulation completed.")

    RUN_META = {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": run_seed,
        "base_seed": SIM_SEED,
        "cycles": CYCLES,
        "run_dir": RUN_DIR.name,
        "policy_name": policy_name,
        "policy_cfg": policy_cfg,
        "model_framing": "empirically grounded scenario-comparison model; outputs are not calibrated forecasts",
        "validation_status": "not calibrated against observed Amsterdam output time series",
        "seasonality": {
            "enabled": bool((np.std(SEASON_REGEN) > 1e-12) or (np.std(SEASON_TRADE) > 1e-12)),
            "SEASON_REGEN": SEASON_REGEN,
            "SEASON_TRADE": SEASON_TRADE,
        },
        "household_representation": {
            "agent_type": "representative household decision unit",
            "biological_demography_enabled": bool(USE_BIOLOGICAL_DEMOGRAPHY),
            "household_size_dynamics_enabled": bool(USE_HOUSEHOLD_SIZE_DYNAMICS),
            "household_size_change_prob": HOUSEHOLD_SIZE_CHANGE_PROB,
            "household_size_need_elasticity": HOUSEHOLD_SIZE_NEED_ELASTICITY,
        },
        "recognition": {
            "reset_initial_recognition": bool(RESET_INITIAL_RECOGNITION),
            "initial_recognition_distribution": f"Beta({INITIAL_RECOG_ALPHA}, {INITIAL_RECOG_BETA})",
            "observability_prob": OBSERVABILITY_PROB,
        },
        "kpi_definitions": {
            "K1": {
                "name": "mean local CE activation",
                "output_columns": ["MEAN_LOCAL_CE_ACTIVATION", "CE_INTENSITY"],
                "definition": (
                    "Spatial average of the grid-cell CE-intensity field. "
                    "The field is updated from local CE flows using local "
                    "normalization and temporal memory."
                ),
            },
            "K2": {
                "name": "flow-based citywide CE index",
                "output_columns": ["CEI_SIM"],
                "definition": (
                    "Weighted citywide index combining normalized delivered CE "
                    "material flows and normalized repair visits. Mean local CE "
                    "activation is excluded to avoid double-counting."
                ),
                "component_columns": ["CE_MATERIAL_FLOW_NORM", "CE_REPAIR_VISIT_NORM"],
                "weights": {
                    "material_flow": CEI_W_TRADE,
                    "repair_visits": CEI_W_REPAIR,
                    "local_activation": CEI_W_INT,
                },
            },
        },
        "k1_spatial_data": {
            "enabled": bool(save_k1_spatial_data),
            "file": str(k1_spatial_npz.name) if k1_spatial_npz is not None else None,
            "last_window": int(k1_spatial_last_window),
            "stored_arrays": [
                "initial_grid",
                "final_grid",
                "change_grid",
                "all_months_mean_grid",
                "all_months_sd_grid",
                "last_window_mean_grid",
                "last_window_temporal_sd_grid",
            ],
        },
        "advantage_field": {
            "raw_min": float(np.nanmin(advantage_raw_field)) if advantage_raw_field is not None else None,
            "raw_max": float(np.nanmax(advantage_raw_field)) if advantage_raw_field is not None else None,
            "normalized_min": float(np.nanmin(advantage_norm_field)) if advantage_norm_field is not None else None,
            "normalized_max": float(np.nanmax(advantage_norm_field)) if advantage_norm_field is not None else None,
            "gini_input": "normalized household spatial-advantage exposure A_h_plus",
        },
        "params": {
            "ENV_WEIGHT": ENV_WEIGHT,
            "SOCIAL_WEIGHT": SOCIAL_WEIGHT,
            "FIN_WEIGHT": FIN_WEIGHT,
            "MOTIVE_CONC": MOTIVE_CONC,
            "NEED": NEED,
            "HUB_TRADES_AS_WEALTH_BONUS": HUB_TRADES_AS_WEALTH_BONUS,
            "EXCLUSION_THRESHOLD": EXCLUSION_THRESHOLD,
            "MAX_PATCH_CAPACITY": MAX_PATCH_CAPACITY,
            "N_HOUSEHOLDS_SCALE": N_HOUSEHOLDS_SCALE,
            "HUB_ACTIVE_FRACTION": HUB_ACTIVE_FRACTION,
            "NEW_HUB_COUNT": NEW_HUB_COUNT,
            "NEW_HUB_NEAR_SECONDARY": NEW_HUB_NEAR_SECONDARY,
            "ADV_HUB_WEIGHT": ADV_HUB_WEIGHT,
            "ADV_SEC_BONUS": ADV_SEC_BONUS,
            "ADV_WASTE_PENALTY": ADV_WASTE_PENALTY,
            "ADV_CONT_WEIGHT": ADV_CONT_WEIGHT,
            "ADV_RECYCL_WEIGHT": ADV_RECYCL_WEIGHT,
            "MOVE_PROB": MOVE_PROB,
            "ETA_KNOW": ETA_KNOW,
            "ETA_OPP": ETA_OPP,
            "ETA_INC": ETA_INC,
            "LAMBDA_PLUS": LAMBDA_PLUS,
            "LAMBDA_MINUS": LAMBDA_MINUS,
            "CE_MEMORY": CE_MEMORY,
            "SAVE_K1_SPATIAL_DATA": bool(save_k1_spatial_data),
            "K1_SPATIAL_LAST_WINDOW": int(k1_spatial_last_window),
            "BULKY_SHARE": BULKY_SHARE,
            "BULKY_THRESHOLD": BULKY_THRESHOLD,
            "BULKY_CE_MULT": BULKY_CE_MULT,
            "BASE_PRIMARY_WAGE": BASE_PRIMARY_WAGE,
            "ALPHA_PRIMARY_DROP": ALPHA_PRIMARY_DROP,
            "PRIMARY_WAGE_FLOOR": PRIMARY_WAGE_FLOOR,
            "BASE_CE_WAGE": BASE_CE_WAGE,
            "CE_WAGE_SLOPE": CE_WAGE_SLOPE,
            "SEASON_REGEN": SEASON_REGEN,
            "SEASON_TRADE": SEASON_TRADE,
            "JOB_CE_SENSITIVITY": JOB_CE_SENSITIVITY,
            "JOB_DISP_SENSITIVITY": JOB_DISP_SENSITIVITY,
            "CEI_W_TRADE": CEI_W_TRADE,
            "CEI_W_REPAIR": CEI_W_REPAIR,
            "CEI_W_INT": CEI_W_INT,
            "UNEMP_REF": UNEMP_REF,
            "UNEMP_BETA_CEI": UNEMP_BETA_CEI,
            "UNEMP_MAX": UNEMP_MAX,
            "CEI_REF": CEI_REF,
            "RISK_UNEMP_MULT": RISK_UNEMP_MULT,
            "GAMMA_NEED_SAVINGS": GAMMA_NEED_SAVINGS,
            "CE_ACTIVE": CE_ACTIVE,
            "REPRO_THRESHOLD": REPRO_THRESHOLD,
            "REPRO_SHARE": REPRO_SHARE,
            "REPRO_PROB": REPRO_PROB,
            "REPRO_MIN_AGE": REPRO_MIN_AGE,
            "REPRO_MAX_AGE": REPRO_MAX_AGE,
        }
    }

    (RUN_DIR / "RUN_META.json").write_text(json.dumps(RUN_META, indent=2, ensure_ascii=False), encoding="utf-8")

    # MASTER metrics CSV
    header = [
        "cycle",
        "D_labor_income",
        "D_adv_exposure",
        "D_adv_cell",
        "D_resources",
        "MEAN_ADV_EXPOSURE",
        "MEAN_RESOURCES",
        "MEAN_LABOR_INCOME",
        "MEAN_HOUSEHOLD_SIZE",
        "TOTAL_SYNTHETIC_POPULATION",
        "D_adv",
        "D_wealth",
        "D_employ",
        "TRADES_COUNT_M",
        "TRADES_MASS_M",
        "TRADES",
        "SORT_DROPOFF_COUNT_M",
        "SORT_DROPOFF_MASS_M",
        "SORT_DROPOFF_MASS_KG",
        "BULKY_DROPOFF_COUNT_M",
        "BULKY_DROPOFF_MASS_M",
        "BULKY_DROPOFF_MASS_KG",
        "REPAIRS_COUNT_M",
        "MOVE_COUNT_M",
        "MOVE_RATE_M",
        "HOUSEHOLD_SIZE_CHANGE_COUNT_M",
        "OBSERVED_CE_ACTIONS_M",
        "CE_ACTION_RECOGNITION_OPPORTUNITIES_M",
        "OBSERVED_CE_ACTION_RATE_M",
        "N_HOUSEHOLDS",
        "MEAN_LOCAL_CE_ACTIVATION",
        "CE_INTENSITY",
        "CE_MATERIAL_FLOW_NORM",
        "CE_REPAIR_VISIT_NORM",
        "CEI_SIM",
        "UNEMPLOYMENT",
        "R_gap",
        "P_voice_corr",
    ]
    with open(RUN_DIR / "MASTER_metrics.csv", "w", newline='', encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        n = len(log["cycle"])
        for i in range(n):
            row = [log.get(h, [np.nan] * n)[i] for h in header]
            w.writerow(row)

    print(f"Outputs ({policy_name}): {RUN_DIR}")
    print("Final household count:", len(households))
    return log

# ---------------- Main ----------------
if __name__ == "__main__":
    print("Policy scenarios to run:", ", ".join(POLICY_SCENARIOS.keys()))
    all_logs = {}

    for pname, pcfg in POLICY_SCENARIOS.items():
        log = run_sim(pname, pcfg)
        all_logs[pname] = log

    COMB_DIR = RUNS_DIR / f"{BASE_RUN_NAME}_combined"
    COMB_DIR.mkdir(parents=True, exist_ok=True)

    scenario_labels = {
        "ce_off": "Linear reference",
        "ce_light": "Low-intensity CE",
        "baseline": "Status-quo CE",
        "ce_strong": "Targeted CE expansion",
        "ce_inclusive_strong": "Inclusive targeted CE",
    }

    scenario_colors = {
        "ce_off":             "black",
        "ce_light":           "tab:gray",
        "baseline":           "tab:blue",
        "ce_medium":          "tab:green",
        "ce_strong":          "tab:orange",
        "ce_inclusive_strong":"tab:red",
    }

    metrics_for_overlay = [
        ("MEAN_LOCAL_CE_ACTIVATION", "Mean local CE activation (K1)"),
        ("CEI_SIM",               "Flow-based citywide CE index (K2)"),
        ("CE_MATERIAL_FLOW_NORM", "Normalized CE material flow"),
        ("CE_REPAIR_VISIT_NORM",  "Normalized repair visits"),
        ("D_resources",           "Gini (accumulated economic resources)"),
        ("D_adv_exposure",        "Gini (normalized spatial-advantage exposure)"),
        ("D_labor_income",        "Gini (monthly labor income)"),
        ("UNEMPLOYMENT",          "Unemployment rate u(t)"),
        ("R_gap",                 "Unemployment gap (risk vs total)"),
        ("P_voice_corr",          "Correlation: voice vs normalized advantage"),
        ("TRADES",                "Total recycling/drop-off flow (kg, monthly)"),
        ("SORT_DROPOFF_COUNT_M",  "Sorting container drop-offs (count, monthly)"),
        ("BULKY_DROPOFF_COUNT_M", "Bulky-waste drop-offs (count, monthly)"),
        ("REPAIRS_COUNT_M",       "Repair Café visits (count, monthly)"),
        ("MOVE_COUNT_M",          "Local exposure updates (count, monthly)"),
    ]

    for key, title in metrics_for_overlay:
        fig, ax = plt.subplots(figsize=(12.0, 8.0))
        fig.subplots_adjust(left=0.11, right=0.98, top=0.90, bottom=0.12)
        any_plotted = False
        for pname, log in all_logs.items():
            if key not in log or not log[key]:
                continue
            cycles = log["cycle"]
            values = log[key]
            lbl = scenario_labels.get(pname, pname)
            col = scenario_colors.get(pname, None)
            ax.plot(cycles, values, label=lbl, linewidth=2.6, color=col)
            any_plotted = True
        if not any_plotted:
            plt.close(fig)
            continue
        ax.set_title(f"{title}\nScenario Comparison", fontsize=22, pad=10)
        ax.set_xlabel("Cycle (Month)", fontsize=18)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=15)
        ax.legend(fontsize=15, framealpha=0.95)
        fig.savefig(COMB_DIR / f"{key}_comparison.pdf", format="pdf", bbox_inches="tight")
        plt.close(fig)

    print(f"All comparative metric figures were saved as PDFs in: {COMB_DIR}")
