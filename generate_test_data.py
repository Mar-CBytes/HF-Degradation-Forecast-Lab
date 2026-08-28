"""
Generate a small synthetic WSPR dataset for testing the pipeline
when no real CSV files are available.

Outage simulation is realistic:
  - Normal operation: short gaps (1-8 min), SNR around -20 dB
  - Pre-outage warning window (up to 30 min before): gaps gradually widen,
    SNR degrades — this is the learnable signal the model targets
  - Outage: silence for 30-90 min, then recovery

Usage:
    python generate_test_data.py
    python run_pipeline.py
"""

from __future__ import annotations

import random
from pathlib import Path
import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

GRIDS = {
    "FN42": (42.5,  -72.5),
    "JO22": (52.0,   12.0),
    "QF22": (-33.5, 151.0),
    "IO91": (51.5,   -2.5),
    "PM74": (34.5,  135.5),
}

BANDS = ["40m", "20m", "30m", "80m"]

random.seed(42)
np.random.seed(42)


def _make_spots_for_month(year: int, month: int) -> pd.DataFrame:
    rows = []
    for grid, (lat, lon) in GRIDS.items():
        t   = pd.Timestamp(year, month, 1, 0, 0, 0, tz="UTC")
        end = t + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)

        # State machine: "normal", "pre_outage", "outage", "recovery"
        state           = "normal"
        outage_start    = None   # when the silence begins
        outage_duration = 0.0    # minutes of silence
        pre_outage_mins = 0.0    # how long the warning ramp lasts
        ramp_elapsed    = 0.0    # minutes spent in pre_outage so far

        while t < end:
            # ── State transitions ─────────────────────────────────────────
            if state == "normal":
                # 3% chance per spot of entering a pre-outage ramp
                if random.random() < 0.03:
                    state           = "pre_outage"
                    pre_outage_mins = random.uniform(15, 30)
                    ramp_elapsed    = 0.0
                gap_min = random.uniform(1, 8)
                snr     = round(np.random.normal(-20, 5), 1)

            elif state == "pre_outage":
                # Ramp: gaps widen 1→20 min, SNR degrades -20→-30 dB
                frac    = min(ramp_elapsed / pre_outage_mins, 1.0)
                gap_min = random.uniform(1, 1 + frac * 19)          # 1–20 min
                snr     = round(np.random.normal(-20 - frac * 10, 3), 1)  # -20 → -30
                gap_min = max(gap_min, 0.5)
                ramp_elapsed += gap_min
                if ramp_elapsed >= pre_outage_mins:
                    state           = "outage"
                    outage_duration = random.uniform(30, 90)
                    outage_start    = t + pd.Timedelta(minutes=gap_min)

            elif state == "outage":
                # Silence: advance time by the full outage duration, emit no spot
                t += pd.Timedelta(minutes=outage_duration)
                state = "recovery"
                continue

            elif state == "recovery":
                # First few spots after outage: shorter gaps, SNR recovering
                gap_min = random.uniform(1, 5)
                snr     = round(np.random.normal(-18, 4), 1)
                state   = "normal"

            t += pd.Timedelta(minutes=gap_min)
            if t >= end:
                break

            rows.append({
                "timestamp":    t.isoformat(),
                "tx_call":      f"W1{grid[:2]}",
                "rx_call":      f"K2{grid[2:]}",
                "snr":          snr,
                "frequency_hz": random.choice([7040000, 14097000, 10140000, 3594000]),
                "tx_grid":      grid,
                "rx_grid":      random.choice(list(GRIDS.keys())),
                "band":         random.choice(BANDS),
                "tx_lat":       lat,
                "tx_lon":       lon,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    for year in [2024, 2025, 2026]:
        for month in range(1, 13):
            if year == 2026 and month > 6:
                break
            df = _make_spots_for_month(year, month)
            fname = RAW_DIR / f"{year}_{month:02d}.csv"
            df.to_csv(fname, index=False)
            print(f"  Written {len(df):,} rows -> {fname.name}")

    print(f"\nDone. {len(list(RAW_DIR.glob('*.csv')))} CSV files in {RAW_DIR}")
