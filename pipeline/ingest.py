"""
Step 1 – Data Ingestion & Unification
--------------------------------------
Reads all WSPR CSV files from data/raw/, normalises column names,
parses timestamps to UTC, converts Maidenhead grid squares to
lat/lon, cleans duplicates and missing values, then saves the
unified dataset to data/processed/wspr_unified.parquet.

Expected CSV column variants (auto-mapped):
  timestamp / date / time / spot_date
  tx_call / txcall / Tx
  rx_call / rxcall / Rx
  snr / SNR
  frequency / freq / MHz
  tx_grid / grid / locator / Locator
  rx_grid / rx_locator
  band / Band
"""

import os
import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import maidenhead as mh

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "wspr_unified.parquet"

# ---------------------------------------------------------------------------
# Column normalisation map  (lower-cased variants → canonical name)
# ---------------------------------------------------------------------------
COL_MAP = {
    # timestamp
    "timestamp": "timestamp", "date": "timestamp", "time": "timestamp",
    "spot_date": "timestamp", "datetime": "timestamp",
    # tx callsign  (real WSPR CSV uses tx_sign; aggregated downloads use tx_sign)
    "tx_call": "tx_call", "txcall": "tx_call", "tx": "tx_call", "call": "tx_call",
    "tx_sign": "tx_call",
    # rx callsign  (real WSPR CSV uses rx_sign; absent in aggregated downloads)
    "rx_call": "rx_call", "rxcall": "rx_call", "rx": "rx_call",
    "rx_sign": "rx_call",
    # rx_count — number of receivers that heard the transmission (aggregated downloads)
    "rx_count": "rx_count",
    # SNR
    "snr": "snr", "snr_db": "snr",
    # frequency
    "frequency": "frequency_hz", "freq": "frequency_hz", "mhz": "frequency_hz",
    "frequency_hz": "frequency_hz",
    # tx grid  (real WSPR CSV uses tx_loc)
    "tx_grid": "tx_grid", "grid": "tx_grid", "locator": "tx_grid",
    "tx_locator": "tx_grid", "maidenhead": "tx_grid",
    "tx_loc": "tx_grid",
    # tx lat/lon  (real WSPR CSV has these directly — no grid conversion needed)
    "tx_lat": "tx_lat", "tx_lon": "tx_lon",
    # rx grid  (real WSPR CSV uses rx_loc)
    "rx_grid": "rx_grid", "rx_locator": "rx_grid",
    "rx_loc": "rx_grid",
    # rx lat/lon
    "rx_lat": "rx_lat", "rx_lon": "rx_lon",
    # band
    "band": "band",
    # power
    "power": "power_dbm", "pwr": "power_dbm",
    # drift
    "drift": "drift",
    # extras present in real data
    "distance": "distance", "azimuth": "azimuth",
    "rx_azimuth": "rx_azimuth", "version": "version", "code": "code",
    "id": "spot_id",
}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns using COL_MAP (case-insensitive)."""
    rename = {}
    for col in df.columns:
        mapped = COL_MAP.get(col.strip().lower())
        if mapped and col != mapped:
            rename[col] = mapped
    df = df.rename(columns=rename)
    # Strip spaces from column names
    df.columns = [c.strip() for c in df.columns]
    # Drop duplicate columns that may arise when multiple source columns
    # (e.g. "date" + "timestamp") both map to the same canonical name.
    # Keep the first occurrence.
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    return df


def _parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Parse timestamp column to UTC-aware datetime."""
    if "timestamp" not in df.columns:
        log.warning("No timestamp column found; skipping timestamp parse.")
        return df

    # Coerce to string first so mixed-type columns are handled uniformly.
    # Use format="mixed" to handle both tz-aware ("2024-01-01T00:03:12+00:00")
    # and tz-naive ("2026-07-01 00:00:00") strings in the same column, then
    # convert to UTC. Falls back to the simpler call on older pandas (<2.0).
    ts_str = df["timestamp"].astype(str)
    try:
        df["timestamp"] = pd.to_datetime(
            ts_str, format="mixed", dayfirst=False, utc=True, errors="coerce"
        )
    except TypeError:
        # pandas < 2.0 does not support format="mixed"
        df["timestamp"] = pd.to_datetime(ts_str, utc=True, errors="coerce")
    bad = df["timestamp"].isna().sum()
    if bad:
        log.warning(f"  Dropped {bad} rows with unparseable timestamps.")
    df = df.dropna(subset=["timestamp"])
    return df


def _grid_to_latlon(grid: str):
    """Convert Maidenhead locator to (lat, lon). Returns (NaN, NaN) on error."""
    try:
        if not isinstance(grid, str) or len(grid) < 4:
            return np.nan, np.nan
        lat, lon = mh.to_location(grid[:6] if len(grid) >= 6 else grid)
        return lat, lon
    except Exception:
        return np.nan, np.nan


def _add_latlon(df: pd.DataFrame) -> pd.DataFrame:
    """Add tx_lat / tx_lon columns.

    If the CSV already contains tx_lat/tx_lon (real WSPR export format),
    use them directly.  Otherwise fall back to Maidenhead grid conversion.
    """
    # Real WSPR CSVs supply tx_lat / tx_lon directly — just ensure numeric
    if "tx_lat" in df.columns and "tx_lon" in df.columns:
        df["tx_lat"] = pd.to_numeric(df["tx_lat"], errors="coerce")
        df["tx_lon"] = pd.to_numeric(df["tx_lon"], errors="coerce")
        log.info("Using tx_lat/tx_lon columns directly from CSV.")
        return df

    # Fall back: derive from Maidenhead grid square
    if "tx_grid" not in df.columns:
        log.warning("tx_grid column missing; lat/lon will not be added.")
        df["tx_lat"] = np.nan
        df["tx_lon"] = np.nan
        return df

    unique_grids = df["tx_grid"].dropna().unique()
    grid_map = {g: _grid_to_latlon(g) for g in unique_grids}
    df["tx_lat"] = df["tx_grid"].map(lambda g: grid_map.get(g, (np.nan, np.nan))[0])
    df["tx_lon"] = df["tx_grid"].map(lambda g: grid_map.get(g, (np.nan, np.nan))[1])
    return df


def _normalise_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure frequency is stored in Hz (float). If values look like MHz, multiply."""
    if "frequency_hz" not in df.columns:
        return df
    df["frequency_hz"] = pd.to_numeric(df["frequency_hz"], errors="coerce")
    # Heuristic: WSPR frequencies are 1–30 MHz; values < 100 are likely in MHz
    mask_mhz = df["frequency_hz"] < 100
    df.loc[mask_mhz, "frequency_hz"] = df.loc[mask_mhz, "frequency_hz"] * 1_000_000
    return df


def _derive_band(df: pd.DataFrame) -> pd.DataFrame:
    """Derive amateur band label from frequency if 'band' column absent."""
    if "band" in df.columns:
        return df
    BAND_EDGES = [
        (1_800_000, 2_000_000, "160m"),
        (3_500_000, 4_000_000, "80m"),
        (5_330_000, 5_410_000, "60m"),
        (7_000_000, 7_300_000, "40m"),
        (10_100_000, 10_150_000, "30m"),
        (14_000_000, 14_350_000, "20m"),
        (18_068_000, 18_168_000, "17m"),
        (21_000_000, 21_450_000, "15m"),
        (24_890_000, 24_990_000, "12m"),
        (28_000_000, 29_700_000, "10m"),
    ]

    def _band(f):
        if pd.isna(f):
            return "unknown"
        for lo, hi, name in BAND_EDGES:
            if lo <= f <= hi:
                return name
        return "unknown"

    if "frequency_hz" in df.columns:
        df["band"] = df["frequency_hz"].apply(_band)
    return df


def _process_single_file(fp: str) -> pd.DataFrame:
    """
    Load, normalise, and clean a single CSV file.
    Returns a tidy DataFrame ready to be written to the unified parquet.
    Processing one file at a time keeps peak RAM usage to ~1 month of data.
    """
    df = pd.read_csv(fp, low_memory=False)
    df["_source_file"] = Path(fp).name
    df = _normalise_columns(df)
    df = _parse_timestamps(df)
    df = _normalise_frequency(df)
    df = _derive_band(df)
    df = _add_latlon(df)

    for col in ["snr", "power_dbm", "drift"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Per-file deduplication on (timestamp, tx_call) — catches duplicate rows
    # within a single month before we ever concatenate across months.
    ts_keys = [c for c in ["timestamp", "tx_call"] if c in df.columns]
    if "spot_id" in df.columns:
        has_id     = df["spot_id"].notna()
        with_id    = df[has_id].drop_duplicates(subset=["spot_id"])
        without_id = df[~has_id].drop_duplicates(subset=ts_keys)
        df = pd.concat([with_id, without_id], ignore_index=True)
    elif ts_keys:
        df = df.drop_duplicates(subset=ts_keys)

    df = df.dropna(subset=["timestamp"])
    return df.sort_values("timestamp")


def load_and_unify(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """
    Load all CSVs from raw_dir one at a time, unify and return a clean
    DataFrame.

    For large datasets (real WSPR downloads, ~300 M rows across 32 months)
    each file is fully processed before the next is read so peak RAM stays
    at roughly one month's worth of data rather than the full dataset.
    The processed frames are then concatenated in one shot; if even that
    exceeds available memory use save_unified_streaming() instead.
    """
    csv_files = sorted(glob.glob(str(raw_dir / "*.csv")))
    if not csv_files:
        log.warning(f"No CSV files found in {raw_dir}. Returning empty DataFrame.")
        return pd.DataFrame()

    frames = []
    total_rows = 0
    for fp in csv_files:
        log.info(f"Reading {fp} …")
        try:
            df = _process_single_file(fp)
            total_rows += len(df)
            frames.append(df)
            log.info(f"  {len(df):,} rows  (running total: {total_rows:,})")
        except Exception as e:
            log.error(f"  Failed to read {fp}: {e}")

    if not frames:
        return pd.DataFrame()

    log.info(f"Concatenating {len(frames)} files ({total_rows:,} rows) …")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    log.info(f"Final unified dataset: {len(combined):,} rows.")
    return combined


def save_unified_streaming(raw_dir: Path = RAW_DIR,
                           output: Path = OUTPUT_FILE) -> Path:
    """
    Memory-efficient alternative to load_and_unify() + save_unified().

    Processes and writes one CSV at a time directly to a parquet file using
    PyArrow incremental writing.  Peak RAM usage is ~1 month of data.
    Use this when the full dataset does not fit in RAM after concat.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    csv_files = sorted(glob.glob(str(raw_dir / "*.csv")))
    if not csv_files:
        log.warning(f"No CSV files found in {raw_dir}.")
        return output

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    total_rows = 0

    for fp in csv_files:
        log.info(f"Reading {fp} …")
        try:
            df = _process_single_file(fp)
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(output), table.schema,
                                          compression="snappy")
            writer.write_table(table)
            total_rows += len(df)
            log.info(f"  {len(df):,} rows written  (total: {total_rows:,})")
            del df, table
        except Exception as e:
            log.error(f"  Failed: {fp}: {e}")

    if writer:
        writer.close()
        log.info(f"Streaming unified dataset saved → {output}  ({total_rows:,} rows)")
    return output


def save_unified(df: pd.DataFrame, output: Path = OUTPUT_FILE) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    log.info(f"Saved unified dataset → {output}")
    return output


if __name__ == "__main__":
    df = load_and_unify()
    if not df.empty:
        save_unified(df)
