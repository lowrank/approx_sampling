#!/usr/bin/env python3
"""
Generate chart data JSON for the interactive mkdocs results page.
Reads curves.json and writes docs/figures/charts_data.json.
"""

import json
import os
import sys

RESULTS_DIR = "results" if len(sys.argv) < 2 else sys.argv[1]
OUT_PATH = "docs/figures/charts_data.json" if len(sys.argv) < 3 else sys.argv[2]

src = os.path.join(RESULTS_DIR, "curves.json")
if not os.path.exists(src):
    print(f"Error: {src} not found. Run an experiment first.")
    sys.exit(1)

with open(src) as f:
    data = json.load(f)

# Keep only a few representative functions to keep file small
keep_fns = [
    "smooth_sinusoid", "smooth_poly", "smooth_exp",
    "oscillatory_high_freq", "oscillatory_chirp", "multiscale",
    "local_gaussians", "local_discontinuity", "boundary_layer", "lorentzian",
    "mixed_osc_decay", "cusp", "wide_bump",
]
out = {}
for fn in keep_fns:
    if fn in data:
        out[fn] = data[fn]

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(out, f)

print(f"Written {OUT_PATH} ({len(out)} functions)")
