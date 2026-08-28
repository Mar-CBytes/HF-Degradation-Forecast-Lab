"""
retrain.py  –  incremental retraining runner
----------------------------------------------
Downloads any missing months of WSPR data, then re-runs the full
pipeline so the model stays current as new spots accumulate.

Typical usage (run monthly from a scheduler):

    python retrain.py                # download any new months, full retrain
    python retrain.py --skip-train   # refresh data only, skip training
    python retrain.py --months 1     # only download the most recent month

Windows Task Scheduler example (monthly, 2nd of month at 03:00):
    Program : C:\Users\stank\Anaconda3\python.exe
    Arguments: C:\Users\stank\.bob\playground\fadewatch\retrain.py
    Start in : C:\Users\stank\.bob\playground\fadewatch

Cron example (monthly):
    0 3 2 * * cd /path/to/fadewatch && python retrain.py >> logs/retrain.log 2>&1
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).resolve().parent / "logs" / "retrain.log"),
    ],
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _last_n_months(n: int) -> tuple[str, str]:
    """Return (start, end) strings for the last N calendar months."""
    from dateutil.relativedelta import relativedelta
    now   = datetime.now(timezone.utc)
    end   = now
    start = now - relativedelta(months=n - 1)
    return f"{start.year}-{start.month:02d}", f"{end.year}-{end.month:02d}"


def main():
    parser = argparse.ArgumentParser(description="FadeWatch incremental retraining")
    parser.add_argument("--months",     type=int, default=None,
                        help="Only download this many recent months (default: all missing)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Download new data but skip model training")
    parser.add_argument("--top-n",      type=int, default=200,
                        help="Top-N transmitters for download filter (default: 200)")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info(f"FadeWatch retrain started  {datetime.now(timezone.utc).isoformat()}")
    log.info("=" * 60)

    # ── Step 1: Download any new/missing months ────────────────────────
    from download_wspr_data import download as wspr_download

    if args.months:
        start, end = _last_n_months(args.months)
        log.info(f"Downloading last {args.months} month(s): {start} → {end}")
    else:
        start = "2024-01"
        end   = None   # defaults to current month
        log.info(f"Downloading all missing months from {start} to present")

    ok, skipped, failed = wspr_download(
        start         = start,
        end           = end,
        top_n         = args.top_n,
        skip_existing = True,   # only fetch months not yet on disk
    )
    log.info(f"Download complete: {ok} new, {skipped} skipped, {failed} failed")

    if ok == 0 and skipped > 0:
        log.info("No new months downloaded — model is already up to date.")
        if not args.skip_train:
            log.info("Use --skip-train=false or delete a CSV to force retrain.")
        return

    # ── Step 2: Run the full pipeline on the new data ─────────────────
    if args.skip_train:
        log.info("--skip-train set; running ingest + features only.")

    from pipeline.ingest  import save_unified_streaming
    from pipeline.label   import run as run_labeling
    from pipeline.solar   import (
        _load_or_fetch_kp, _load_or_fetch_sfi,
        _load_or_fetch_xray, _load_or_fetch_gfz_kp,
    )
    from pipeline.features import run as run_features
    from pipeline.split    import split_and_save
    from pipeline.train    import train

    base          = Path(__file__).resolve().parent
    unified_file  = base / "data" / "processed" / "wspr_unified.parquet"
    labeled_file  = base / "data" / "processed" / "wspr_labeled.parquet"
    features_file = base / "data" / "processed" / "wspr_features.parquet"

    log.info("Stage 1: Ingestion (streaming) …")
    save_unified_streaming(output=unified_file)

    log.info("Stage 2: Outage labeling …")
    run_labeling(unified_file, labeled_file)

    log.info("Stage 3: Solar data refresh …")
    _load_or_fetch_gfz_kp()
    _load_or_fetch_kp()
    _load_or_fetch_sfi()
    _load_or_fetch_xray()

    log.info("Stage 4: Feature engineering …")
    run_features(labeled_file, features_file)

    log.info("Stage 5: Train/val/test split …")
    split_and_save(features_file)

    if not args.skip_train:
        log.info("Stage 6: Model training …")
        rf, schema = train()
        log.info(
            f"Retrain complete. "
            f"Test F1={schema['evaluation']['test_f1']:.4f}  "
            f"ROC-AUC={schema['evaluation']['roc_auc']:.4f}"
        )
    else:
        log.info("Skipping training (--skip-train).")

    log.info(f"Retrain finished  {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    main()
