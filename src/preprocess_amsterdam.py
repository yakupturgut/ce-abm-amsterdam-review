#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
preprocess_amsterdam.py

Builds the processed Amsterdam grid and initial household file used by abm.py.
Run this only when the raw/external data or preprocessing assumptions change.

Main outputs:
    data/processed/amsterdam_grid.npz
    data/processed/init_agents.pkl
    data/processed/PREP_SUMMARY.json
"""
from pathlib import Path
from collections import defaultdict, Counter
import warnings, random, pickle, csv, json
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.features as rfeat
import matplotlib.pyplot as plt
import re as _re

# ---------------- Paths ----------------
# Project root is inferred from this file location, so the code can run
# after cloning/downloading the repository without hard-coded local paths.
ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw"
EXTERNAL_DATA = ROOT / "data" / "external"
PROCESSED_DATA = ROOT / "data" / "processed"
PROCESSED_DATA.mkdir(parents=True, exist_ok=True)

GEOJSON        = RAW_DATA / "wijken.geojson"
CONTAINERS     = RAW_DATA / "waste_containers.csv"
CAFES_CSV      = EXTERNAL_DATA / "repair_cafes_amsterdam.csv"
WASTE_XLSX     = RAW_DATA / "amsterda_waste_patches.xlsx"
SECONDARY_XLSX = RAW_DATA / "amsterdam_secondary_patches.xlsx"

# BBGA files; the helper searches for flexible file names
BBGA_DEFS_CANDIDATES = [
    RAW_DATA / "bbga_indicatoren_definities.csv",
    RAW_DATA / "BBGA_Indicatoren_definities.csv",
]
BBGA_VALS_CANDIDATES = [
    RAW_DATA / "bbga_kerncijfers.csv",
    RAW_DATA / "BBGA_kerncijfers.csv",
    RAW_DATA / "bbga_kerncijfers_*.csv",
    RAW_DATA / "kerncijfers*.csv",
]

OUT_NPZ = PROCESSED_DATA / "amsterdam_grid.npz"
OUT_PKL = PROCESSED_DATA / "init_agents.pkl"

# ---------------- Params ----------------
GRID = 60
PRIMARY_N = 250
DESIRED_TOTAL = 4000
SEED = 0
# Recycling-centre / secondary patch placement settings
SECONDARY_BASE_STOCK = (70, 140)  # initial stock (U[min,max])
SECONDARY_REGEN_RATE = 2.0        # secondary regeneration rate used by the grid layer
SECONDARY_MAX_R      = 12         # search radius for resolving occupied cells
SECONDARY_TARGET     = 90         # minimum target number of secondary cells after seeding clusters

# Agent init params
AGENT_BASE_WEALTH = 35.0  # Default initial resource level if BBGA income is unavailable
AGENT_BASE_VOICE  = 0.7   # Base political influence
AGENT_REC_BETA    = (2, 5) # Beta(2,5) for initial recognition (skewed low)

# Household-size distribution for representative household decision units.
HOUSEHOLD_SIZE_VALUES = np.array([1, 2, 3, 4, 5, 6], dtype=int)
HOUSEHOLD_SIZE_PROBS  = np.array([0.38, 0.34, 0.13, 0.09, 0.04, 0.02], dtype=float)
HOUSEHOLD_SIZE_PROBS  = HOUSEHOLD_SIZE_PROBS / HOUSEHOLD_SIZE_PROBS.sum()

VERSION = "preprocess_v13.1"
CRS_TAG = "EPSG:28992"

random.seed(SEED); np.random.seed(SEED)
np.seterr(all='ignore')
warnings.filterwarnings("ignore", category=FutureWarning)

# Patch codes used in the grid array
EMPTY, PRIMARY, SECONDARY, WASTE, HUB = 0, 1, 2, 3, 4

info = lambda m: print(f"[•] {m}")

# ---------------- Helpers ----------------
def try_import_pyproj():
    try:
        from pyproj import Transformer
        return Transformer
    except Exception:
        info("  Warning: 'pyproj' is not installed. WGS84 to RD conversion may fail if needed.")
        return None

Transformer = try_import_pyproj()

def to_rd_from_wgs84(lons, lats):
    """WGS84 (lon,lat) -> RD New (x,y)."""
    if Transformer is None:
        raise RuntimeError("The 'pyproj' package is required to convert WGS84 coordinates to Dutch RD coordinates.")
    tf = Transformer.from_crs("EPSG:4326","EPSG:28992", always_xy=True)
    xr, yr = zip(*tf.itransform(zip(lons, lats)))
    return pd.Series(xr, dtype=float), pd.Series(yr, dtype=float)

def nearest_empty(ptype, i0, j0, max_r, GRID):
    """Return the nearest EMPTY grid cell when the requested cell is occupied."""
    if ptype[i0, j0] == EMPTY:
        return (i0, j0)
    for r in range(1, max_r+1):
        for i in range(max(0, i0-r), min(GRID, i0+r+1)):
            for j in range(max(0, j0-r), min(GRID, j0+r+1)):
                if abs(i-i0) + abs(j-j0) == r and ptype[i, j] == EMPTY:
                    return (i, j)
    return None

def read_csv_smart(path: Path) -> pd.DataFrame:
    """Read a CSV file while guessing separator and text encoding."""
    encodings = ["utf-8", "utf-8-sig", "latin1"]
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors="replace") as f:
                sample = f.read(4096)
                dialect = csv.Sniffer().sniff(sample, delimiters=",;|\t")
                sep = dialect.delimiter
            return pd.read_csv(path, encoding=enc, sep=sep)
        except Exception:
            continue
    return pd.read_csv(path)

def find_first_existing(candidates):
    for c in candidates:
        if "*" in str(c):
            matches = list(c.parent.glob(c.name))
            if matches:
                return matches[0]
        elif c.exists():
            return c
    return None

def _norm_code(s: str) -> str:
    """Normalize BBGA/GeoJSON area codes: uppercase and keep only letters/numbers."""
    s = str(s).strip().upper()
    s = _re.sub(r'[^A-Z0-9]', '', s)
    return s

def sample_household_size() -> int:
    """Draw a neutral initial household size independent of income."""
    return int(np.random.choice(HOUSEHOLD_SIZE_VALUES, p=HOUSEHOLD_SIZE_PROBS))

def neutral_initial_recognition() -> float:
    """Draw initial recognition independently of income/neighborhood affluence."""
    return float(np.clip(np.random.beta(*AGENT_REC_BETA), 0.0, 1.0))

# ---------------- 1) Neighborhood raster ----------------
info("Reading neighborhood GeoJSON file...")
wijk = gpd.read_file(GEOJSON).to_crs(28992)
code_candidates = ["Gebiedcode15","Gebiedcode","CBS_Wijkcode","Wijkcode","code"]
code_col = next((c for c in code_candidates if c in wijk.columns), None)
if code_col is None:
    raise ValueError(f"No suitable neighborhood-code column was found. Available columns: {list(wijk.columns)}")

xmin, ymin, xmax, ymax = wijk.total_bounds
cell_w, cell_h = (xmax - xmin) / GRID, (ymax - ymin) / GRID
aff = rasterio.transform.from_origin(xmin, ymax, cell_w, cell_h)

grid_idx = np.full((GRID, GRID), -1, dtype=int)
rfeat.rasterize(((geom, i) for i, geom in enumerate(wijk.geometry)),
                out=grid_idx, transform=aff, fill=-1)

# Map each grid cell to a normalized neighborhood code
wijk_codes_raw = wijk[code_col].astype(str).reset_index(drop=True).to_numpy()
wijk_codes_norm = np.array([_norm_code(c) for c in wijk_codes_raw], dtype=object)
cell_code  = np.where(grid_idx >= 0, wijk_codes_norm[grid_idx], "")

# ---------------- 2) Patch matrisleri ----------------
ptype = np.zeros((GRID, GRID), dtype=int)
stock = np.zeros_like(ptype, dtype=float)
regen = np.zeros_like(ptype, dtype=float)

# ---------------- 3) CE infrastructure: sorting containers, recycling centres and repair cafés ----------------
container_cells = []   # sorting-container locations
recycling_cells = []   # amsterdam_secondary_patches (siyah elmas)
hub_cells       = []   # repair-café / hub locations
hub_fraction    = {}   # label stored for each repair hub

# 3a) Sorting containers (waste_containers.csv) → SECONDARY patch + marker
if CONTAINERS.exists():
    info("Sorting container (waste_containers.csv) okunuyor...")
    try:
        cols = pd.read_csv(CONTAINERS, nrows=1).columns
        geom_col = next((c for c in cols if "geom" in c.lower() or "eometrie" in c.lower()), None)
        if geom_col is None:
            raise ValueError("No geometry column was found in waste_containers.csv.")
        cont = pd.read_csv(CONTAINERS, usecols=[geom_col])

        pat_pt = _re.compile(r"POINT\s*\(\s*([-\d\.,]+)\s+([-\d\.,]+)\s*\)", _re.I)
        xy = cont[geom_col].astype(str).str.extract(pat_pt).replace(",",".", regex=True).astype(float)
        valid = xy[0].notna() & xy[1].notna()
        x_idx = ((xy.loc[valid, 0] - xmin) / cell_w).round().clip(0, GRID-1).astype(int)
        y_idx = ((ymax - xy.loc[valid, 1]) / cell_h).round().clip(0, GRID-1).astype(int)

        for ix, iy in zip(x_idx, y_idx):
            x, y = int(ix), int(iy)
            if 0 <= x < GRID and 0 <= y < GRID:
                container_cells.append((x, y))
                # Convert to a secondary patch unless the cell is already waste or a hub
                if ptype[x, y] in (EMPTY, PRIMARY):
                    ptype[x, y] = SECONDARY
                    if stock[x, y] <= 0:
                        stock[x, y] = np.random.uniform(*SECONDARY_BASE_STOCK)
                    regen[x, y] = SECONDARY_REGEN_RATE
        info(f"  -> Processed {len(container_cells)} sorting-container cells.")
    except Exception as e:
        info(f"  Warning: waste_containers.csv could not be read: {e}")
else:
    info("waste_containers.csv was not found; no sorting-container markers will be added.")

# 3b) Repair cafés from repair_cafes_amsterdam.csv become HUB patches and firms
if CAFES_CSV.exists():
    info("Reading Repair Café CSV...")
    cafes = pd.read_csv(CAFES_CSV)
    cafes.columns = cafes.columns.str.strip().str.lower()
    if {"x_rd","y_rd"}.issubset(cafes.columns):
        xr, yr = cafes["x_rd"].astype(float), cafes["y_rd"].astype(float)
    elif {"lat","lng"}.issubset(cafes.columns) or {"lat","lon"}.issubset(cafes.columns):
        lon_col = "lng" if "lng" in cafes.columns else "lon"
        xr, yr = to_rd_from_wgs84(cafes[lon_col], cafes["lat"])
    else:
        xr = yr = pd.Series(dtype=float)
        info("  Warning: coordinate columns for Repair Café locations were not found; skipping them.")

    for xrd, yrd in zip(xr, yr):
        ix = int((xrd - xmin) / cell_w)
        iy = int((ymax - yrd) / cell_h)
        if 0 <= ix < GRID and 0 <= iy < GRID:
            hub_cells.append((ix, iy))
            ptype[ix, iy] = HUB
            hub_fraction[(ix, iy)] = "RepairCafe"
    info(f"  -> Processed {len(hub_cells)} repair-café / hub cells.")
else:
    info("Repair Café CSV was not found; no repair hubs were added.")

# ---------------- 4) Waste hotspots (amsterda_waste_patches.xlsx) ----------------
waste_cells = []
if WASTE_XLSX.exists():
    info("Reading waste-hotspot patch file...")
    wdf = pd.read_excel(WASTE_XLSX)
    wdf.columns = wdf.columns.str.strip().str.lower()
    if {"x_rd","y_rd"}.issubset(wdf.columns):
        xr, yr = wdf["x_rd"].astype(float), wdf["y_rd"].astype(float)
    elif {"lat","lon"}.issubset(wdf.columns) or {"lat","lng"}.issubset(wdf.columns):
        lon_col = "lon" if "lon" in wdf.columns else "lng"
        xr, yr = to_rd_from_wgs84(wdf[lon_col], wdf["lat"])
    else:
        xr = yr = pd.Series(dtype=float)
        info("  Warning: coordinate columns for waste hotspots were not found; skipping them.")

    for x_rd, y_rd in zip(xr, yr):
        ix = int((x_rd - xmin) / cell_w)
        iy = int((ymax - y_rd) / cell_h)
        if 0 <= ix < GRID and 0 <= iy < GRID:
            ptype[ix, iy] = WASTE   # waste hotspots override other patch types
            stock[ix, iy] = 0.0
            regen[ix, iy] = 0.0
            waste_cells.append((ix, iy))
    info(f"  -> Added {len(waste_cells)} waste-hotspot cells from file.")
else:
    info("Waste patch file was not found; three random waste cells will be used.")
    empties = [(i, j) for i in range(GRID) for j in range(GRID)]
    waste_cells = random.sample(empties, min(3, len(empties)))
    for x, y in waste_cells:
        ptype[x, y] = WASTE

# ---------------- 5) SECONDARY patch'ler (recycling centres + cluster) ----------------
secondary_cells = []
if SECONDARY_XLSX.exists():
    info("Secondary patch (recycling centres) XLSX okunuyor...")
    sdf = pd.read_excel(SECONDARY_XLSX)
    sdf.columns = sdf.columns.str.strip().str.lower()

    if {"x_rd","y_rd"}.issubset(sdf.columns):
        xr, yr = sdf["x_rd"].astype(float), sdf["y_rd"].astype(float)
    elif {"lat","lon"}.issubset(sdf.columns) or {"lat","lng"}.issubset(sdf.columns):
        lon_col = "lon" if "lon" in sdf.columns else "lng"
        xr, yr = to_rd_from_wgs84(sdf[lon_col], sdf["lat"])
    else:
        xr = yr = pd.Series(dtype=float)
        info("  Warning: coordinate columns for recycling centres were not found; skipping them.")

    # Convert coordinates to grid cells and remove duplicates
    sec_idx = []
    for x_rd, y_rd in zip(xr, yr):
        ix = int((x_rd - xmin) / cell_w)
        iy = int((ymax - y_rd) / cell_h)
        if 0 <= ix < GRID and 0 <= iy < GRID:
            sec_idx.append((ix, iy))
    sec_idx = list(set(sec_idx))
    info(f"  -> Unique recycling-centre cells mapped to the grid before conflict resolution: {len(sec_idx)}")

    placed = set()
    for ix, iy in sec_idx:
        if 0 <= ix < GRID and 0 <= iy < GRID:
            # Do not overwrite HUB/WASTE cells; move to the nearest empty cell if needed
            if ptype[ix, iy] in (HUB, WASTE):
                alt = nearest_empty(ptype, ix, iy, SECONDARY_MAX_R, GRID)
                if alt is None:
                    continue
                spot = alt
            else:
                spot = (ix, iy)

            x, y = spot
            if (x, y) in placed:
                continue

            ptype[x, y] = SECONDARY
            regen[x, y] = SECONDARY_REGEN_RATE
            stock[x, y] = np.random.uniform(*SECONDARY_BASE_STOCK)
            secondary_cells.append((x, y))
            recycling_cells.append((x, y))
            placed.add((x, y))

    # Seed nearby cells until the target number of secondary patches is reached
    if len(secondary_cells) < SECONDARY_TARGET:
        candidates = set()
        neigh = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]
        for (x, y) in list(secondary_cells):
            for dx, dy in neigh:
                i, j = x+dx, y+dy
                if 0 <= i < GRID and 0 <= j < GRID and ptype[i, j] in (EMPTY, PRIMARY):
                    candidates.add((i, j))
        
        need = SECONDARY_TARGET - len(secondary_cells)
        if need > 0 and candidates:
            picks = random.sample(list(candidates), min(need, len(candidates)))
            for (i, j) in picks:
                ptype[i, j] = SECONDARY
                regen[i, j] = SECONDARY_REGEN_RATE
                stock[i, j] = np.random.uniform(*SECONDARY_BASE_STOCK)
                secondary_cells.append((i, j))

    info(f"  -> Final secondary cells after conflict resolution and seeding: {len(secondary_cells)}")
else:
    info("Secondary patch file was not found; no recycling-centre markers will be added.")

# 5b) Sorting containers were already marked as SECONDARY; fill stock/regeneration if needed
for (x, y) in container_cells:
    if ptype[x, y] != WASTE and ptype[x, y] != HUB:
        ptype[x, y] = SECONDARY
        if stock[x, y] <= 0:
            stock[x, y] = np.random.uniform(*SECONDARY_BASE_STOCK)
        if regen[x, y] <= 0:
            regen[x, y] = SECONDARY_REGEN_RATE
        if (x, y) not in secondary_cells:
            secondary_cells.append((x, y))

# ---------------- 6) PRIMARY (secondary'den SONRA!) ----------------
# Kalan EMPTY'leri PRIMARY yap
empt = [(i, j) for i in range(GRID) for j in range(GRID) if ptype[i, j] == EMPTY]
random.shuffle(empt)
for (x, y) in empt[:PRIMARY_N]:
    ptype[x, y] = PRIMARY
    stock[x, y] = np.random.uniform(50, 100)
    regen[x, y] = 1.0

# ---------------- 7) (Opsiyonel) BBGA ----------------
def_path = find_first_existing(BBGA_DEFS_CANDIDATES)
val_path = find_first_existing(BBGA_VALS_CANDIDATES)
bbga_ok = False
pop = pd.Series(dtype=float)
inc = pd.Series(dtype=float)

try:
    if not def_path or not val_path:
        raise FileNotFoundError("BBGA definitions or core-indicators file was not found.")

    defs = read_csv_smart(def_path)
    vals = read_csv_smart(val_path)

    defs.columns = defs.columns.str.strip()
    vals.columns = vals.columns.str.strip()

    def _code(kw):
        m = defs.loc[defs["Label"].astype(str).str.contains(kw, case=False, na=False), "Variabele"]
        if m.empty: return None
        return m.iloc[0]

    pop_code = _code("inwoner")    # population
    inc_code = _code("inkomen")    # gelir
    if not pop_code or not inc_code:
        raise KeyError("Could not identify the population or income indicator codes in BBGA files.")

    def pick(cols, *cand):
        for c in cand:
            if c in cols: return c
        return None

    id_col = pick(vals.columns, "Indicatordefinitieid", "IndicatorDefinitieId", "Indicator", "Variabele")
    gc_col = pick(vals.columns, "Gebiedcode15", "Gebiedscode15", "Gebiedcode", "Wijkcode")
    val_col = pick(vals.columns, "Waarde", "waarde", "Value")
    
    pop_s = vals[vals[id_col] == pop_code].groupby(gc_col)[val_col].last()
    inc_s = vals[vals[id_col] == inc_code].groupby(gc_col)[val_col].last()

    def _to_num(s):
        return pd.to_numeric(pd.Series(s, dtype=str).str.replace(",", ".", regex=False), errors="coerce")

    pop = _to_num(pop_s).astype(float)
    inc = _to_num(inc_s).astype(float)

    pop.index = pop.index.map(_norm_code)
    inc.index = inc.index.map(_norm_code)
    
    bbga_ok = (not pop.empty and not inc.empty)
    if bbga_ok:
        info(f"BBGA verisi okundu. (def={def_path.name}, vals={val_path.name})")
    else:
        info("  Warning: BBGA values look empty; population/income matching may have failed.")
except Exception as e:
    info(f"  Warning: BBGA data could not be read ({e}); default households will be used.")

# ---------------- 8) Initialize household agents from BBGA-based population weights ----------------
info("Distributing household agents using BBGA data or defaults...")
init_agents = []

if not bbga_ok:
    info(f"  → BBGA yok/eksik: {DESIRED_TOTAL} rasgele ajan.")
    candidates = [(i, j) for i in range(GRID) for j in range(GRID) if ptype[i, j] != WASTE]
    for _ in range(DESIRED_TOTAL):
        x, y = random.choice(candidates)
        wealth = np.random.uniform(AGENT_BASE_WEALTH, AGENT_BASE_WEALTH*2)
        voice_base = AGENT_BASE_VOICE
        recognition = neutral_initial_recognition()
        household_size = sample_household_size()
        init_agents.append((x, y, wealth, voice_base, recognition, 0, household_size))
else:
    cells_by_code = defaultdict(list)
    for i in range(GRID):
        for j in range(GRID):
            c = cell_code[i, j]
            if c and c in pop.index and ptype[i, j] != WASTE:
                cells_by_code[c].append((i, j))

    usable_codes = {code: cells for code, cells in cells_by_code.items() if code in inc.index and cells}

    if not usable_codes:
        info("  Warning: BBGA codes did not match GeoJSON codes; falling back to random placement.")
        candidates = [(i, j) for i in range(GRID) for j in range(GRID) if ptype[i, j] != WASTE]
        for _ in range(DESIRED_TOTAL):
            x, y = random.choice(candidates)
            wealth = np.random.uniform(AGENT_BASE_WEALTH, AGENT_BASE_WEALTH*2)
            voice_base = AGENT_BASE_VOICE
            recognition = np.random.beta(*AGENT_REC_BETA) * AGENT_BASE_VOICE
            init_agents.append((x, y, wealth, voice_base, recognition, 0))
    else:
        w = pd.Series({code: pop.get(code, 0.0) for code in usable_codes})
        w = w.clip(lower=0.0); target = DESIRED_TOTAL
        if w.sum() == 0: w[:] = 1.0
        raw = (w / w.sum() * target).fillna(0.0)
        alloc = raw.astype(int)
        remainder = int(target - alloc.sum())
        if remainder > 0:
            frac = (raw - alloc).sort_values(ascending=False)
            for code in frac.index[:remainder]:
                alloc[code] += 1
        
        inc_mean = inc[inc.notna()].mean() if not inc.empty else 1000.0
        
        for code, n in alloc.items():
            cells = usable_codes[code]
            income = inc.get(code, inc_mean)
            ln_mu = np.log(max(1, income) / inc_mean * AGENT_BASE_WEALTH)
            ln_sigma = 0.45 

            for _ in range(int(n)):
                x, y = random.choice(cells)
                wealth = np.random.lognormal(ln_mu, ln_sigma)
                # Recognition is intentionally initialized from a neutral distribution,
                # not from neighborhood income. This avoids imposing an income-based
                # recognition advantage at t=0.
                recognition = neutral_initial_recognition()
                household_size = sample_household_size()
                voice_base = AGENT_BASE_VOICE 
                init_agents.append((x, y, wealth, voice_base, recognition, 0, household_size))

info(f"  -> Created {len(init_agents)} initial household agents using BBGA-based allocation.")

# ---------------- 9) Save preprocessing artefacts ----------------
np.savez_compressed(
    OUT_NPZ,
    ptype=ptype,
    stock=stock,
    regen=regen,
    hub_cells=np.array(hub_cells, dtype=int) if hub_cells else np.empty((0,2), dtype=int),
    waste_cells=np.array(waste_cells, dtype=int) if waste_cells else np.empty((0,2), dtype=int),
    container_cells=np.array(container_cells, dtype=int) if container_cells else np.empty((0,2), dtype=int),
    recycling_cells=np.array(recycling_cells, dtype=int) if recycling_cells else np.empty((0,2), dtype=int),
    hub_fracs=np.array([(x, y, frac) for (x, y), frac in (hub_fraction.items() if hub_cells else [])],
                       dtype=object),
    cell_code=cell_code,
    crs=CRS_TAG,
    version=VERSION
)

with open(OUT_PKL, "wb") as f:
    pickle.dump(init_agents, f)

# ---------------- 10) Quick preprocessing diagnostics ----------------
# quick map (debug)
plt.figure(figsize=(6, 5))
show = np.zeros_like(ptype, int)
sx, sy = np.where(ptype == SECONDARY); show[sx, sy] = 2
wx, wy = zip(*waste_cells) if waste_cells else ([],[])
for x,y in waste_cells: show[x,y] = 3
hx, hy = zip(*hub_cells) if hub_cells else ([],[])
for x,y in hub_cells: show[x,y] = 4
plt.imshow(show.T, origin="lower")
plt.title("Secondary (2), Waste (3), Hubs (4)")
plt.xticks([]); plt.yticks([])
plt.tight_layout(); plt.savefig(PROCESSED_DATA / "prep_quickmap.png", dpi=140); plt.close()

# agent density
counts = np.zeros((GRID, GRID), dtype=int)
for x, y, *_ in init_agents:
    counts[x, y] += 1
plt.figure(figsize=(6, 5))
plt.imshow(counts.T, origin="lower")
plt.title("Initial household density (BBGA weighted)")
plt.xticks([]); plt.yticks([])
plt.colorbar(fraction=0.046, pad=0.02)
plt.tight_layout(); plt.savefig(PROCESSED_DATA / "prep_agent_density.png", dpi=140); plt.close()

# ---------------- 11) Save preprocessing summary JSON ----------------
hub_frac_hist = Counter(hub_fraction.values()) if hub_cells else Counter()
summary = {
    "version": VERSION,
    "crs": CRS_TAG,
    "grid": GRID,
    "seed": SEED,
    "patch_counts": {
        "EMPTY": int(np.sum(ptype == EMPTY)),
        "PRIMARY": int(np.sum(ptype == PRIMARY)),
        "SECONDARY": int(np.sum(ptype == SECONDARY)),
        "WASTE": int(np.sum(ptype == WASTE)),
        "HUB": int(np.sum(ptype == HUB)),
    },
    "hubs": len(hub_cells),
    "wastes": len(waste_cells),
    "containers": len(container_cells),
    "recycling_centres": len(recycling_cells),
    "hub_fraction_hist": dict(hub_frac_hist),
    "agents_created": len(init_agents),
    "initial_wealth_mean": float(np.mean([a[2] for a in init_agents])),
    "initial_recog_mean": float(np.mean([a[4] for a in init_agents])),
    "initial_household_size_mean": float(np.mean([a[6] for a in init_agents])) if init_agents and len(init_agents[0]) >= 7 else None,
    "recognition_initialization": "Neutral Beta distribution independent of income",
    "bbga": {
        "pop_nonempty": bool(bbga_ok),
        "income_used": "LogNormal (BBGA Weighted)" if bbga_ok else "Uniform (Fallback)"
    }
}
(Path(PROCESSED_DATA / "PREP_SUMMARY.json")).write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("Preprocessing completed.")
