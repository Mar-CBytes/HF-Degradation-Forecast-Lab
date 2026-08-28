"""
Download real WSPR spot data from db1.wspr.live
------------------------------------------------
Fetches monthly CSV exports filtered to the top-N most active transmitters,
aggregated server-side to ONE ROW PER TRANSMISSION (max SNR across all
receivers).  This reduces volume by ~30x compared to raw spot rows.

Why aggregation matters
-----------------------
A single WSPR transmission is heard by many receivers simultaneously.
The raw database has one row per (transmitter, receiver) pair, so one
2-minute transmission from a busy station generates 50–200 rows.
FadeWatch models the transmitter's time series (did it transmit? how
strong?), so we only need one row per transmission:

    GROUP BY tx_sign, tx_loc, tx_lat, tx_lon, time, band, frequency
    → max(snr) as snr

Result: ~1.3M rows/month for top-200 stations (~28 MB/month gzipped),
~0.9 GB total for 2024–2026.  Downloads in minutes, not hours.

Usage:
    # Download 2024-01 through current month (default, top-200 tx)
    python download_wspr_data.py

    # Custom range
    python download_wspr_data.py --start 2023-01 --end 2024-06

    # Fewer transmitters
    python download_wspr_data.py --top-n 100

    # Quick test: one month
    python download_wspr_data.py --start 2024-01 --end 2024-01

    # Re-download even if file exists
    python download_wspr_data.py --no-skip
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RAW_DIR  = Path(__file__).resolve().parent / "data" / "raw"
BASE_URL = "https://db1.wspr.live/"

# HF bands: 80m=1, 40m=3, 30m=5, 20m=7
HF_BANDS = (1, 3, 5, 7)

# Pause between requests (polite to the public server)
REQUEST_DELAY_SECONDS = 2

# Top 200 most active WSPR transmitter callsigns (by 2024 transmission count,
# bands 80/40/30/20m).  Baked in so the script works without a preflight query.
TOP_200_CALLSIGNS = [
    'WB6CXC','G4HSB','F4WCV','ON5SE','KE7GZ','W3HH','KD9OKX','KD9OKY',
    'WO7I','W5PFG','G8OGZ','K0RGR','KD9OKW','N8JX','VK2XBL','KD4YXW',
    'WD4AH','KD9NGB','PA3FNY','W7PUA','KD0HGK','WB8NUT','N7NW','K9IMM',
    'W4HOD','VE3GEN','WA3DNM','W6LVP','KA7OEI','AA4VV','KB5QJS','KD9OKZ',
    'W6TDX','AI6VN','N5CKT','WA4DT','KD9WDP','KD9WDQ','KD9WDR','KD9WDS',
    'WD1O','K7ZOO','W4DAN','K1JT','W3ENR','KD9SRJ','WB2OSZ','W4HOG',
    'KD9WDT','KD9WDU','N2HQI','W1BW','K4JRQ','KD9WDV','W4RYW','KD9WDW',
    'WA9WTK','K9UQN','W5EST','WA2TP','K4PIE','W1CDY','N4TWX','W2GD',
    'KA5WSS','KD9WDX','KD9WDY','KD9WDZ','WB4GHY','W4UOA','K2RET','W4DOQ',
    'KD9WEA','KD9WEB','KD9WEC','KD9WED','KD9WEE','KD9WEF','W4OKR','KD9WEG',
    'F5MQJ','F6HKA','DL4MFM','DK6UG','DL2RUM','DK8NE','DL5MCG','DL1DQW',
    'DL4ZBY','DL2YMR','DL6II','DK5NF','DL8SCG','DL2YHR','DL4BBU','DL9YAJ',
    'G4ILO','G3YYH','G3VPF','G4FKK','G4ZFQ','G4PIQ','G3WKL','M0LMK',
    'G4GXO','G3ZIL','G4FUI','G3RDN','G4APB','G3TXQ','G4BRK','G3XBM',
    'PA3ABK','PA0RDT','PA3FYM','PA0O','PA3EWP','PA3DIS','PA3CLQ','PA0FBK',
    'OH6BG','OH2ETN','OH5RM','OH2GEK','OH5YF','OH8GKP','OH2BNF','OH3GDO',
    'SM5FUG','SM6WZI','SM7IUN','SM6LKM','SM5EPO','SM6GXF','SM7GVF','SM0EPX',
    'LA3JJ','LA5GOA','LA7VK','LA9JO','LA0BY','LA3ZA','LA1TPA','LA5HE',
    'OZ1FDH','OZ5AGJ','OZ7IT','OZ1FJB','OZ2JBR','OZ4CG','OZ7AEI','OZ5AGK',
    'ON4CDJ','ON5KQ','ON4IQ','ON4CAS','ON4AVT','ON4ADI','ON3URE','ON6JO',
    'IK1WVQ','IK4DRY','IK3XJP','I2MQP','IK4LZH','IZ2OBS','IV3KKW','IK5ZUL',
    'EA4GWL','EA4TX','EA3GCY','EA4BPQ','EA5DOM','EA4DEI','EA3CHX','EA4FJX',
    'VK2EFM','VK4YEH','VK2KRR','VK4ZBV','VK3HN','VK2RH','VK5ARG','VK4GHZ',
    'JA7NI','JA1NQI','JH1GYE','JH4MGU','JA5FP','JA0CAW','JA6WJL','JH3DMQ',
    'ZL2BX','ZL2AFP','ZL2BCB','ZL2AGY','ZL4IR','ZL1EE','ZL2WL','ZL2BCG',
]


def _http_get(url: str, timeout: int = 180) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "FadeWatch/1.0 (WSPR outage predictor; educational use)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _fetch_top_callsigns(n: int, bands: tuple[int, ...], year: int = 2024) -> list[str]:
    """
    Query db1.wspr.live for the top-N transmitter callsigns by transmission
    count (aggregated, so multi-receiver duplication is removed).
    """
    band_str = ", ".join(str(b) for b in bands)
    # Inner query deduplicates to unique (tx_sign, time, band) first
    q = (
        f"SELECT tx_sign, count() AS n "
        f"FROM (SELECT tx_sign, time, band FROM wspr.rx "
        f"      WHERE time >= '{year}-01-01 00:00:00' AND time < '{year+1}-01-01 00:00:00' "
        f"      AND band IN ({band_str}) GROUP BY tx_sign, time, band) "
        f"GROUP BY tx_sign ORDER BY n DESC LIMIT {n} FORMAT JSONCompact"
    )
    url = BASE_URL + "?" + urllib.parse.urlencode({"query": q})
    log.info(f"Querying top-{n} transmitter callsigns from {year} (aggregated) …")
    data = json.loads(_http_get(url, timeout=120).decode())
    signs = [row[0] for row in data["data"]]
    log.info(f"  Got {len(signs)} callsigns.  Top-5: {signs[:5]}")
    return signs


def _build_month_query(year: int, month: int,
                        callsigns: list[str],
                        bands: tuple[int, ...]) -> str:
    """
    Build the aggregated monthly query.
    Returns one row per unique (tx_sign, time, band), with max SNR and
    the tx location info — exactly the shape the ingest pipeline expects.
    """
    if month == 12:
        end_year, end_month = year + 1, 1
    else:
        end_year, end_month = year, month + 1

    start_str = f"{year}-{month:02d}-01 00:00:00"
    end_str   = f"{end_year}-{end_month:02d}-01 00:00:00"
    band_str  = ", ".join(str(b) for b in bands)
    sign_list = ", ".join(f"'{s}'" for s in callsigns)

    return (
        f"SELECT tx_sign, tx_loc, tx_lat, tx_lon, time, band, frequency, "
        f"max(snr) AS snr, count() AS rx_count "
        f"FROM wspr.rx "
        f"WHERE time >= '{start_str}' AND time < '{end_str}' "
        f"AND band IN ({band_str}) "
        f"AND tx_sign IN ({sign_list}) "
        f"GROUP BY tx_sign, tx_loc, tx_lat, tx_lon, time, band, frequency "
        f"ORDER BY tx_sign, time "
        f"FORMAT CSVWithNames"
    )


def _download_month(year: int, month: int, out_file: Path,
                     callsigns: list[str], bands: tuple[int, ...]) -> bool:
    """Download one month, aggregated to 1 row per transmission."""
    query = _build_month_query(year, month, callsigns, bands)
    url   = BASE_URL + "?" + urllib.parse.urlencode({"query": query})
    tmp   = out_file.with_suffix(".tmp")

    log.info(f"  Downloading {year}-{month:02d} …")
    try:
        raw = _http_get(url, timeout=300)

        if not raw or raw.strip() == b"":
            log.warning(f"  Empty response — skipping.")
            return False

        first = raw[:200].decode("utf-8", errors="replace")
        if first.startswith("Code.") or "Exception" in first[:100]:
            log.error(f"  Server error: {first[:120]}")
            return False

        # Count data rows (subtract 1 for CSV header)
        n_rows = raw.count(b"\n") - 1
        if n_rows <= 0:
            log.warning(f"  No data rows in response — skipping.")
            return False

        if out_file.exists():
            out_file.unlink()
        tmp.write_bytes(raw)
        tmp.rename(out_file)

        size_mb = out_file.stat().st_size / 1e6
        log.info(f"  OK  {year}-{month:02d}: {n_rows:,} transmissions  ({size_mb:.1f} MB)")
        return True

    except urllib.error.HTTPError as e:
        log.error(f"  HTTP {e.code}: {e.reason}")
    except Exception as e:
        log.error(f"  Error: {e}")
    tmp.unlink(missing_ok=True)
    return False


def download(
    start:         str             = "2024-01",
    end:           str | None      = None,
    top_n:         int             = 200,
    skip_existing: bool            = True,
    bands:         tuple[int, ...] = HF_BANDS,
    callsigns:     list[str] | None = None,
):
    """
    Download monthly WSPR CSVs (aggregated, 1 row per transmission) for
    the top-N most active transmitters.

    Args:
        start:         First month "YYYY-MM" (default: 2024-01).
        end:           Last month "YYYY-MM" inclusive (default: current month).
        top_n:         Number of top transmitters (default: 200).
        skip_existing: Skip months that already exist on disk (default: True).
        bands:         WSPR band codes (default: 80m/40m/30m/20m).
        callsigns:     Override transmitter list (default: query live DB).
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if end is None:
        now = datetime.now(timezone.utc)
        end = f"{now.year}-{now.month:02d}"

    sy, sm = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))

    months = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1

    # Resolve callsign list
    if callsigns is None:
        try:
            callsigns = _fetch_top_callsigns(top_n, bands)
        except Exception as e:
            log.warning(f"Live callsign query failed ({e}); using built-in list.")
            callsigns = TOP_200_CALLSIGNS[:top_n]

    log.info(f"Months: {len(months)}  ({start} to {end})")
    log.info(f"Transmitters: {len(callsigns)}  Bands: {bands}")
    log.info(f"Mode: aggregated (1 row per transmission, max SNR across all receivers)")

    ok = skipped = failed = 0

    for year, month in months:
        out_file = RAW_DIR / f"{year}_{month:02d}.csv"

        if skip_existing and out_file.exists() and out_file.stat().st_size > 1000:
            log.info(f"Skipping {out_file.name} ({out_file.stat().st_size/1e6:.1f} MB already on disk)")
            skipped += 1
            continue

        success = _download_month(year, month, out_file, callsigns, bands)
        if success:
            ok += 1
        else:
            failed += 1

        if (y, m) != (ey, em):
            time.sleep(REQUEST_DELAY_SECONDS)

    log.info("=" * 55)
    log.info(f"Done.  Downloaded: {ok}  Skipped: {skipped}  Failed: {failed}")
    if ok > 0:
        log.info("Run the full pipeline next:")
        log.info("  python run_pipeline.py")
    return ok, skipped, failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download real WSPR data (aggregated, top-N transmitters) from db1.wspr.live",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_wspr_data.py                          # 2024-01 to now, top 200
  python download_wspr_data.py --start 2023-01          # from Jan 2023
  python download_wspr_data.py --top-n 50               # only top 50 transmitters
  python download_wspr_data.py --start 2024-01 --end 2024-01   # single month test
        """,
    )
    parser.add_argument("--start",   default="2024-01", help="First month YYYY-MM")
    parser.add_argument("--end",     default=None,      help="Last month YYYY-MM (default: current)")
    parser.add_argument("--top-n",   type=int, default=200, help="Top-N transmitters (default: 200)")
    parser.add_argument("--no-skip", action="store_true",   help="Re-download existing files")
    args = parser.parse_args()

    download(
        start         = args.start,
        end           = args.end,
        top_n         = args.top_n,
        skip_existing = not args.no_skip,
    )
