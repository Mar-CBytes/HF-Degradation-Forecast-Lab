"""
Step 3 – Outage Labeling
--------------------------
Defines an outage at a grid-square location and produces:

  outage_label(t)        = 1 if an outage IS occurring at time t
  future_outage_label(t) = 1 if an outage STARTS within HORIZON_MINUTES

Outage definition (any condition triggers):
  - Time gap between consecutive spots > GAP_THRESHOLD_MINUTES
  - Rolling SNR drop below SNR_THRESHOLD for SUSTAINED_SPOTS spots in a row
  - No spots received for SILENCE_WINDOW_MINUTES

All thresholds are tunable via constants below.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
INPUT_FILE    = PROCESSED_DIR / "wspr_unified.parquet"
OUTPUT_FILE   = PROCESSED_DIR / "wspr_labeled.parquet"

# ---------------------------------------------------------------------------
# Outage-definition thresholds
# ---------------------------------------------------------------------------
GAP_THRESHOLD_MINUTES   = 30   # gap between consecutive spots to flag outage
SNR_THRESHOLD           = -28  # dB — sustained low SNR signals degradation
SUSTAINED_SPOTS         = 3    # how many consecutive below-threshold SNR spots
SILENCE_WINDOW_MINUTES  = 30   # rolling window (min) with zero spots → outage
HORIZON_MINUTES         = 60   # how far ahead to look for future outage label


def _label_single_grid(df_grid: pd.DataFrame) -> pd.DataFrame:
    """
    Given all rows for ONE grid square (sorted by timestamp),
    compute time_gap, outage_label, and future_outage_label.
    """
    df_grid = df_grid.sort_values("timestamp").copy()
    df_grid = df_grid.reset_index(drop=True)

    # ── 1. Time-gap outage ───────────────────────────────────────────────
    df_grid["time_gap_min"] = (
        df_grid["timestamp"].diff().dt.total_seconds() / 60
    )
    gap_outage = df_grid["time_gap_min"] > GAP_THRESHOLD_MINUTES

    # ── 2. Sustained low-SNR outage ─────────────────────────────────────
    if "snr" in df_grid.columns:
        below = (df_grid["snr"] < SNR_THRESHOLD).astype(int)
        rolling_low = below.rolling(SUSTAINED_SPOTS, min_periods=SUSTAINED_SPOTS).sum()
        snr_outage = rolling_low >= SUSTAINED_SPOTS
    else:
        snr_outage = pd.Series(False, index=df_grid.index)

    # ── 3. Silence-window outage ─────────────────────────────────────────
    ts = df_grid.set_index("timestamp")
    spot_counts = ts.resample("1min").size().rename("spot_count")
    rolling_spots = spot_counts.rolling(f"{SILENCE_WINDOW_MINUTES}min", min_periods=1).sum()
    silence_bins = rolling_spots[rolling_spots == 0].index
    silence_outage = df_grid["timestamp"].isin(silence_bins)

    # ── Combine ───────────────────────────────────────────────────────────
    df_grid["outage_label"] = (gap_outage | snr_outage | silence_outage).astype(int)

    # ── Future outage label  (vectorised — no Python loop) ────────────────
    # For each row i, future_outage_label=1 if:
    #   - the row is NOT currently in an outage (outage_label=0), AND
    #   - an outage_label=1 row exists in the half-open window
    #     (timestamp[i], timestamp[i] + HORIZON_MINUTES]
    #
    # Excluding current-outage rows is critical: without it, outage rows
    # label themselves as "future outage=1" (distance 0), which makes the
    # majority of positive labels mid-outage rather than pre-outage, and
    # the model learns to detect ongoing outages instead of predicting them.
    horizon_ns = np.timedelta64(HORIZON_MINUTES, "m")
    ts_ns   = df_grid["timestamp"].values.astype("datetime64[ns]")
    out_arr = df_grid["outage_label"].values.astype(bool)

    # Indices of outage rows
    outage_idx = np.where(out_arr)[0]

    if len(outage_idx) == 0:
        df_grid["future_outage_label"] = 0
    else:
        outage_times = ts_ns[outage_idx]
        # For each row, find the index of the next outage time strictly after ts
        next_outage_pos = np.searchsorted(outage_times, ts_ns, side="right")
        # Clip to valid range for safe indexing
        next_outage_pos_clipped = np.clip(next_outage_pos, 0, len(outage_times) - 1)
        next_outage_time = outage_times[next_outage_pos_clipped]
        # Future outage exists if:
        #   1. there is a next outage (pos is in bounds), AND
        #   2. it falls within the horizon window, AND
        #   3. this row is not itself an outage row (pre-outage only)
        future_outage = (
            (next_outage_pos < len(outage_times)) &
            (next_outage_time - ts_ns <= horizon_ns) &
            ~out_arr
        )
        df_grid["future_outage_label"] = future_outage.astype(int)

    return df_grid


def label_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply outage labeling to every grid square independently."""
    if "tx_grid" not in df.columns:
        log.warning("tx_grid column not found; labeling on whole dataset as one group.")
        return _label_single_grid(df)

    results = []
    grids = df["tx_grid"].dropna().unique()
    log.info(f"Labeling outages for {len(grids)} grid squares …")

    for grid in grids:
        subset = df[df["tx_grid"] == grid].copy()
        labeled = _label_single_grid(subset)
        results.append(labeled)

    combined = pd.concat(results, ignore_index=True).sort_values("timestamp")
    n_out = combined["outage_label"].sum()
    n_fut = combined["future_outage_label"].sum()
    pct_out = 100 * n_out / max(len(combined), 1)
    pct_fut = 100 * n_fut / max(len(combined), 1)
    log.info(f"outage_label=1: {n_out:,} rows ({pct_out:.1f}%)")
    log.info(f"future_outage_label=1: {n_fut:,} rows ({pct_fut:.1f}%)")
    return combined


def run(input_file: Path = INPUT_FILE, output_file: Path = OUTPUT_FILE):
    df = pd.read_parquet(input_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    labeled = label_dataset(df)
    labeled.to_parquet(output_file, index=False)
    log.info(f"Labeled dataset saved → {output_file}")
    return labeled


if __name__ == "__main__":
    run()
