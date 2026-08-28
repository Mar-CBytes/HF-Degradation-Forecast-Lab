"""
run_pipeline.py  –  end-to-end pipeline runner
------------------------------------------------
Runs all pipeline stages in order:
  1. Ingest       raw CSVs → wspr_unified.parquet
  2. Label        outage events
  3. Solar        fetch/cache Kp + SFI from NOAA SWPC
  4. Features     engineer features (includes solar join)
  5. Split        chronological train/val/test
  6. Train        Random Forest model

Run:
    python run_pipeline.py [--skip-ingest] [--skip-train]
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    parser = argparse.ArgumentParser(description="WSPR Outage Predictor – full pipeline runner")
    parser.add_argument("--skip-ingest",  action="store_true", help="Skip CSV ingestion (use existing unified file)")
    parser.add_argument("--skip-train",   action="store_true", help="Skip model training")
    args = parser.parse_args()

    from pipeline.ingest   import save_unified_streaming
    from pipeline.label    import run as run_labeling
    from pipeline.solar    import (
        _load_or_fetch_kp, _load_or_fetch_sfi,
        _load_or_fetch_xray, _load_or_fetch_gfz_kp,
    )
    from pipeline.features import run as run_features
    from pipeline.split    import split_and_save
    from pipeline.train    import train

    processed_dir = Path(__file__).resolve().parent / "data" / "processed"
    unified_file  = processed_dir / "wspr_unified.parquet"
    labeled_file  = processed_dir / "wspr_labeled.parquet"
    features_file = processed_dir / "wspr_features.parquet"

    # ── Stage 1: Ingest ────────────────────────────────────────────────
    if not args.skip_ingest:
        log.info("─── Stage 1: Data Ingestion ─────────────────────────────")
        # Use streaming writer: processes one CSV at a time, never loads
        # the full 300M-row dataset into RAM simultaneously.
        result = save_unified_streaming(output=unified_file)
        if not unified_file.exists() or unified_file.stat().st_size == 0:
            log.error("No data ingested. Place CSV files in data/raw/ and retry.")
            sys.exit(1)
    else:
        log.info("Skipping ingestion (--skip-ingest).")

    if not unified_file.exists():
        log.error(f"Unified file not found: {unified_file}")
        sys.exit(1)

    # ── Stage 2: Label ─────────────────────────────────────────────────
    log.info("─── Stage 2: Outage Labeling ────────────────────────────────")
    run_labeling(unified_file, labeled_file)

    # ── Stage 3: Solar/geomagnetic data ────────────────────────────────
    log.info("─── Stage 3: Solar/Geomagnetic Data ─────────────────────────")
    log.info("Fetching GFZ Potsdam historical Kp archive (may take ~30 s on first run) …")
    _load_or_fetch_gfz_kp()   # historical Kp backfill (cached weekly)
    log.info("Fetching NOAA SWPC: Kp, SFI, GOES X-ray …")
    _load_or_fetch_kp()        # 3-day 1-min Kp
    _load_or_fetch_sfi()       # daily F10.7
    _load_or_fetch_xray()      # 6-hour GOES X-ray

    # ── Stage 4: Features ─────────────────────────────────────────────
    log.info("─── Stage 4: Feature Engineering ───────────────────────────")
    run_features(labeled_file, features_file)

    # ── Stage 5: Split ─────────────────────────────────────────────────
    log.info("─── Stage 5: Time-aware Split ───────────────────────────────")
    split_and_save(features_file)

    # ── Stage 6: Train ─────────────────────────────────────────────────
    if not args.skip_train:
        log.info("─── Stage 6: Model Training ─────────────────────────────")
        rf, schema = train()
        log.info(f"Training done. Test F1={schema['evaluation']['test_f1']:.4f}")
    else:
        log.info("Skipping training (--skip-train).")

    log.info("Pipeline complete. Start the API with:")
    log.info("  uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload")


if __name__ == "__main__":
    main()
