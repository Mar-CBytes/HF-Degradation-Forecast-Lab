"""
Step 2 – Time-Aware Data Splitting
------------------------------------
Splits the unified dataset chronologically (no shuffling):
  Train : earliest 60% of the timeline
  Val   : next    20%
  Test  : final   20%

Also provides per-location splits using the same date boundaries.

Usage:
    python -m pipeline.split
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
INPUT_FILE = PROCESSED_DIR / "wspr_unified.parquet"
SPLIT_DIR = PROCESSED_DIR / "splits"

TRAIN_FRAC = 0.60
VAL_FRAC = 0.20
# TEST_FRAC = remaining 20%


def compute_split_boundaries(df: pd.DataFrame):
    """Return (train_end, val_end) timestamps that honour the 60/20/20 split."""
    t_min = df["timestamp"].min()
    t_max = df["timestamp"].max()
    total_seconds = (t_max - t_min).total_seconds()

    train_end = t_min + pd.Timedelta(seconds=total_seconds * TRAIN_FRAC)
    val_end   = t_min + pd.Timedelta(seconds=total_seconds * (TRAIN_FRAC + VAL_FRAC))
    return train_end, val_end


def split_dataframe(df: pd.DataFrame):
    """Split df chronologically. Returns (train, val, test, meta_dict)."""
    train_end, val_end = compute_split_boundaries(df)

    train = df[df["timestamp"] <= train_end].copy()
    val   = df[(df["timestamp"] > train_end) & (df["timestamp"] <= val_end)].copy()
    test  = df[df["timestamp"] > val_end].copy()

    meta = {
        "global_start":     str(df["timestamp"].min()),
        "train_end":        str(train_end),
        "val_end":          str(val_end),
        "global_end":       str(df["timestamp"].max()),
        "train_rows":       len(train),
        "val_rows":         len(val),
        "test_rows":        len(test),
    }

    if "outage_label" in df.columns:
        meta["train_outages"] = int(train["outage_label"].sum())
        meta["val_outages"]   = int(val["outage_label"].sum())
        meta["test_outages"]  = int(test["outage_label"].sum())

    return train, val, test, meta


def split_and_save(input_file: Path = INPUT_FILE, split_dir: Path = SPLIT_DIR):
    """Load unified file, split, save parquet shards, print documentation."""
    df = pd.read_parquet(input_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    split_dir.mkdir(parents=True, exist_ok=True)

    train, val, test, meta = split_dataframe(df)

    train.to_parquet(split_dir / "train.parquet", index=False)
    val.to_parquet(split_dir / "val.parquet",   index=False)
    test.to_parquet(split_dir / "test.parquet",  index=False)

    log.info("=" * 60)
    log.info("SPLIT DOCUMENTATION")
    log.info("=" * 60)
    log.info(f"  Dataset range  : {meta['global_start']}  →  {meta['global_end']}")
    log.info(f"  Train   (60 %) : up to  {meta['train_end']}   ({meta['train_rows']:,} rows)")
    log.info(f"  Val     (20 %) : up to  {meta['val_end']}     ({meta['val_rows']:,} rows)")
    log.info(f"  Test    (20 %) : after  {meta['val_end']}      ({meta['test_rows']:,} rows)")
    if "train_outages" in meta:
        log.info(f"  Outage events  → train:{meta['train_outages']}  val:{meta['val_outages']}  test:{meta['test_outages']}")
    log.info("=" * 60)

    # Save meta as JSON for downstream reference
    import json
    with open(split_dir / "split_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return train, val, test, meta


# ── Convenience accessor ────────────────────────────────────────────────────
def load_splits(split_dir: Path = SPLIT_DIR):
    """Load pre-saved train/val/test parquets. Returns (train, val, test)."""
    return (
        pd.read_parquet(split_dir / "train.parquet"),
        pd.read_parquet(split_dir / "val.parquet"),
        pd.read_parquet(split_dir / "test.parquet"),
    )


def get_split_boundaries(split_dir: Path = SPLIT_DIR):
    """Return (train_end, val_end) as pd.Timestamp from saved meta."""
    import json
    with open(split_dir / "split_meta.json") as f:
        meta = json.load(f)
    return (
        pd.Timestamp(meta["train_end"]),
        pd.Timestamp(meta["val_end"]),
    )


if __name__ == "__main__":
    split_and_save()
