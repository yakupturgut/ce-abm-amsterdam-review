#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
data_extraction.py

Optional utility for downloading Repair Cafe locations for Amsterdam and
saving them as an external input file. This script requires internet access and
is not needed for routine replicated analysis if data/external already exists.
"""
# repair_cafes_amsterdam_bbox.py
from pathlib import Path
import requests, pandas as pd
from pyproj import Transformer

# 1) API call with Amsterdam bounding-box parameters
NE = "52.4300,5.0700"
SW = "52.2800,4.7300"
url = (f"https://www.repaircafe.org/wp-json/v1/map?"
       f"northeast={NE}&southwest={SW}")

data = requests.get(url, timeout=30).json()          # The API returns a single JSON list of repair-cafe records

# 2) Convert API response to a tabular DataFrame
df = pd.DataFrame(data)

# 3) Convert WGS84 coordinates to Dutch RD coordinates (EPSG:28992)
tf = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
df[['x_rd','y_rd']] = df['coordinate'].str.split(',', expand=True).astype(float) \
                       .apply(lambda s: pd.Series(tf.transform(s[1], s[0])), axis=1)

# 4) Save the output CSV used by preprocess_amsterdam.py
cols = ['name','address','coordinate','x_rd','y_rd','external_link']
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "external"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "repair_cafes_amsterdam.csv"

df[cols].to_csv(OUT_CSV, index=False, encoding="utf-8")

print(f"Saved {len(df)} Repair Café records to {OUT_CSV}")

