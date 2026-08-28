"""
Solar & Geomagnetic Data  –  pipeline/solar.py
------------------------------------------------
Fetches Kp index, F10.7 solar flux index (SFI), and GOES X-ray flux
from NOAA Space Weather Prediction Center (SWPC) public JSON APIs.
No API key required.

APIs used:
  Kp (1-min, 3-day)    https://services.swpc.noaa.gov/json/planetary_k_index_1m.json
  SFI (daily)          https://services.swpc.noaa.gov/json/f107_index.json
  X-ray flux (5-min)   https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json
  SWPC alerts          https://services.swpc.noaa.gov/products/alerts.json

Historical Kp backfill:
  GFZ Potsdam publishes the definitive Kp index as a plain-text file
  covering the full geomagnetic record. The 1-hour resolution file is
  fetched once and cached; it is used to backfill the WSPR training set
  so that training rows get real historical Kp values instead of the
  quiet-day default (Kp=2) that the 3-day NOAA endpoint cannot provide.

  GFZ URL: https://www.gfz-potsdam.de/fileadmin/gfz/sec32/Kp_ap_Ap_SN_F107_since_1932.txt

Public functions:
  fetch_kp()                 → pd.DataFrame  [timestamp (UTC), kp]
  fetch_sfi()                → pd.DataFrame  [date, sfi]
  fetch_xray()               → pd.DataFrame  [timestamp (UTC), xray_class (int 0-4)]
  fetch_gfz_kp()             → pd.DataFrame  [timestamp (UTC), kp]  (historical, hourly)
  fetch_alerts()             → list[dict]    (raw SWPC alert dicts)
  merge_solar_data(df)       → df  with kp_index, kp_trend, kp_max_3h,
                                       geomagnetic_storm, sfi, kp_x_sfi,
                                       xray_flux_class
  get_current_solar()        → dict  (live values for inference)
  get_current_alerts()       → list[dict]  (active/recent SWPC alerts)
"""

from __future__ import annotations

import io
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────
DATA_DIR        = Path(__file__).resolve().parents[1] / "data" / "processed"
KP_CACHE        = DATA_DIR / "solar_kp_cache.parquet"
SFI_CACHE       = DATA_DIR / "solar_sfi_cache.parquet"
XRAY_CACHE      = DATA_DIR / "solar_xray_cache.parquet"
GFZ_KP_CACHE    = DATA_DIR / "solar_gfz_kp_cache.parquet"
CACHE_MAX_AGE   = 24 * 3600      # seconds before refreshing short-lived caches
GFZ_CACHE_AGE   = 7 * 24 * 3600  # GFZ file is updated monthly — weekly refresh is fine

# ── NOAA endpoints ────────────────────────────────────────────────────────
KP_URL     = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
# NOTE: the f107_index.json endpoint was retired; use the daily text product instead.
SFI_URL    = "https://services.swpc.noaa.gov/text/daily-solar-indices.txt"
XRAY_URL   = "https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json"
ALERTS_URL = "https://services.swpc.noaa.gov/products/alerts.json"

# ── GFZ Potsdam – definitive Kp index (hourly, since 1932) ───────────────
# The file is a fixed-width text format; we parse the columns we need.
GFZ_KP_URL = (
    "https://kp.gfz.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt"
)

# ── X-ray class encoding ─────────────────────────────────────────────────
# GOES long-wavelength (0.1–0.8 nm) flux bands:
#   A: < 1e-7 W/m²   → 0
#   B: 1e-7–1e-6     → 1
#   C: 1e-6–1e-5     → 2
#   M: 1e-5–1e-4     → 3
#   X: ≥ 1e-4        → 4
XRAY_THRESHOLDS = [1e-7, 1e-6, 1e-5, 1e-4]  # boundaries between classes

# Safe fallbacks
KP_DEFAULT        = 2.0
SFI_DEFAULT       = 120.0
XRAY_DEFAULT      = 0    # A-class (quiet)

# ── In-process inference cache (5-minute TTL) ─────────────────────────────
_live_cache: dict = {}
_alerts_cache: dict = {}
_LIVE_TTL    = 300   # seconds
_ALERTS_TTL  = 600   # 10 minutes — alerts change slowly


# ---------------------------------------------------------------------------
# X-ray flux helpers
# ---------------------------------------------------------------------------

def _encode_xray_flux(flux_wm2: float) -> int:
    """Convert GOES long-wavelength flux (W/m²) to integer class 0–4."""
    if flux_wm2 < XRAY_THRESHOLDS[0]:
        return 0  # A
    if flux_wm2 < XRAY_THRESHOLDS[1]:
        return 1  # B
    if flux_wm2 < XRAY_THRESHOLDS[2]:
        return 2  # C
    if flux_wm2 < XRAY_THRESHOLDS[3]:
        return 3  # M
    return 4      # X


# ---------------------------------------------------------------------------
# Raw fetchers
# ---------------------------------------------------------------------------

def fetch_kp(timeout: int = 10) -> pd.DataFrame:
    """
    Download the NOAA 3-day 1-minute Kp index.
    Returns DataFrame with columns [timestamp (UTC, tz-aware), kp].
    """
    resp = requests.get(KP_URL, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    rows = []
    for entry in data:
        ts_raw = entry.get("time_tag") or entry.get("timestamp") or entry.get("time")
        kp_raw = entry.get("kp_index") or entry.get("Kp") or entry.get("kp")
        if ts_raw is None or kp_raw is None:
            continue
        try:
            ts = pd.Timestamp(ts_raw, tz="UTC")
            kp = float(kp_raw)
            rows.append({"timestamp": ts, "kp": kp})
        except (ValueError, TypeError):
            continue
    if not rows:
        raise ValueError("NOAA Kp response contained no usable rows.")
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def fetch_sfi(timeout: int = 10) -> pd.DataFrame:
    """
    Download the NOAA daily F10.7 solar flux index (10.7 cm radio flux).

    Parses the NOAA SWPC daily solar indices text product:
      https://services.swpc.noaa.gov/text/daily-solar-indices.txt

    Columns in the file (space-separated, after the header block):
      YYYY MM DD  F10.7  Sunspot  ...
    F10.7 is the second data column (index 3 in 0-based split).

    Returns DataFrame with columns [date (datetime, UTC), sfi (float)].
    """
    resp = requests.get(SFI_URL, timeout=timeout)
    resp.raise_for_status()
    rows = []
    for line in resp.text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith(":"):
            continue
        parts = line.split()
        # Data lines begin with YYYY MM DD
        if len(parts) < 4:
            continue
        try:
            year  = int(parts[0])
            month = int(parts[1])
            day   = int(parts[2])
            sfi   = float(parts[3])
            if sfi <= 0:
                continue
            date = pd.Timestamp(year=year, month=month, day=day, tz="UTC")
            rows.append({"date": date, "sfi": sfi})
        except (ValueError, IndexError):
            continue
    if not rows:
        raise ValueError("NOAA daily solar indices text contained no usable SFI rows.")
    df = pd.DataFrame(rows)
    return df.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def fetch_xray(timeout: int = 10) -> pd.DataFrame:
    """
    Download GOES primary X-ray flux (6-hour window, 5-min cadence).
    Returns DataFrame with columns [timestamp (UTC, tz-aware), xray_flux (float),
    xray_flux_class (int 0-4)].

    Only the long-wavelength channel (0.1–0.8 nm) is used; this is the
    channel relevant to HF radio absorption (D-layer heating).
    """
    resp = requests.get(XRAY_URL, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    rows = []
    for entry in data:
        ts_raw   = entry.get("time_tag")
        flux_raw = entry.get("flux")
        energy   = entry.get("energy", "")
        # Long-wavelength channel: "0.1-0.8nm" or similar label
        if energy and "0.1" not in str(energy):
            continue
        if ts_raw is None or flux_raw is None:
            continue
        try:
            ts   = pd.Timestamp(ts_raw, tz="UTC")
            flux = float(flux_raw)
            if flux <= 0:
                continue
            rows.append({
                "timestamp":       ts,
                "xray_flux":       flux,
                "xray_flux_class": _encode_xray_flux(flux),
            })
        except (ValueError, TypeError):
            continue
    if not rows:
        raise ValueError("GOES X-ray response contained no usable rows.")
    df = pd.DataFrame(rows).sort_values("timestamp")
    return df.drop_duplicates("timestamp").reset_index(drop=True)


def fetch_gfz_kp(timeout: int = 60) -> pd.DataFrame:
    """
    Download the GFZ Potsdam definitive Kp index (hourly, since 1932).

    The file is a large fixed-width text file (~3 MB). We parse only the
    columns needed: year, month, day, and the 8 three-hourly Kp values
    per day.  Each value is converted to a UTC timestamp at the start of
    its 3-hour interval.

    Returns DataFrame [timestamp (UTC, tz-aware), kp].
    """
    resp = requests.get(GFZ_KP_URL, timeout=timeout)
    resp.raise_for_status()
    text = resp.text

    rows = []
    for line in text.splitlines():
        line = line.strip()
        # Skip header/comment lines
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        # GFZ format (space-separated, fixed width):
        # YYYY MM DD days days_m Bsr dB Kp1 Kp2 Kp3 Kp4 Kp5 Kp6 Kp7 Kp8 ap1..ap8 Ap SN F10.7obs F10.7adj D
        # Kp values are at positions 7–14 (0-indexed), after the 7 header cols
        if len(parts) < 15:
            continue
        try:
            year  = int(parts[0])
            month = int(parts[1])
            day   = int(parts[2])
            # 8 three-hourly Kp values starting at index 7
            kp_vals = [float(parts[7 + i]) for i in range(8)]
            # Skip rows with missing data flags
            if any(v < 0 for v in kp_vals):
                continue
        except (ValueError, IndexError):
            continue
        for i, kp in enumerate(kp_vals):
            hour = i * 3
            try:
                ts = pd.Timestamp(year=year, month=month, day=day, hour=hour, tz="UTC")
                rows.append({"timestamp": ts, "kp": kp})
            except Exception:
                continue

    if not rows:
        raise ValueError("GFZ Kp file contained no parseable rows.")

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    log.info(f"GFZ Kp: {len(df):,} rows  {df['timestamp'].min().date()} – {df['timestamp'].max().date()}")
    return df


def fetch_alerts(timeout: int = 10) -> list[dict]:
    """
    Download the NOAA SWPC alerts/watches/warnings JSON feed.
    Returns the raw list of alert dicts (may be empty on network error).
    """
    resp = requests.get(ALERTS_URL, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_fresh(path: Path, max_age: int = CACHE_MAX_AGE) -> bool:
    if not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) < max_age


def _load_or_fetch_kp() -> pd.DataFrame:
    if _cache_fresh(KP_CACHE):
        return pd.read_parquet(KP_CACHE)
    try:
        df = fetch_kp()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(KP_CACHE, index=False)
        log.info(f"Kp cache refreshed → {KP_CACHE}  ({len(df)} rows)")
        return df
    except Exception as e:
        log.warning(f"Could not fetch Kp from NOAA ({e}); using cache or defaults.")
        if KP_CACHE.exists():
            return pd.read_parquet(KP_CACHE)
        now = pd.Timestamp.now(tz="UTC")
        return pd.DataFrame({"timestamp": [now], "kp": [KP_DEFAULT]})


def _load_or_fetch_sfi() -> pd.DataFrame:
    if _cache_fresh(SFI_CACHE):
        return pd.read_parquet(SFI_CACHE)
    try:
        df = fetch_sfi()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(SFI_CACHE, index=False)
        log.info(f"SFI cache refreshed → {SFI_CACHE}  ({len(df)} rows)")
        return df
    except Exception as e:
        log.warning(f"Could not fetch SFI from NOAA ({e}); using cache or defaults.")
        if SFI_CACHE.exists():
            return pd.read_parquet(SFI_CACHE)
        today = pd.Timestamp.now(tz="UTC").normalize()
        return pd.DataFrame({"date": [today], "sfi": [SFI_DEFAULT]})


def _load_or_fetch_xray() -> pd.DataFrame:
    if _cache_fresh(XRAY_CACHE):
        return pd.read_parquet(XRAY_CACHE)
    try:
        df = fetch_xray()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(XRAY_CACHE, index=False)
        log.info(f"X-ray cache refreshed → {XRAY_CACHE}  ({len(df)} rows)")
        return df
    except Exception as e:
        log.warning(f"Could not fetch X-ray from NOAA ({e}); using cache or defaults.")
        if XRAY_CACHE.exists():
            return pd.read_parquet(XRAY_CACHE)
        now = pd.Timestamp.now(tz="UTC")
        return pd.DataFrame({
            "timestamp":       [now],
            "xray_flux":       [1e-8],
            "xray_flux_class": [XRAY_DEFAULT],
        })


def _load_or_fetch_gfz_kp() -> pd.DataFrame:
    """Load historical GFZ Kp, refreshing at most weekly."""
    if _cache_fresh(GFZ_KP_CACHE, max_age=GFZ_CACHE_AGE):
        return pd.read_parquet(GFZ_KP_CACHE)
    try:
        df = fetch_gfz_kp()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(GFZ_KP_CACHE, index=False)
        log.info(f"GFZ Kp cache saved → {GFZ_KP_CACHE}  ({len(df):,} rows)")
        return df
    except Exception as e:
        log.warning(f"Could not fetch GFZ Kp ({e}); falling back to NOAA short-term cache.")
        if GFZ_KP_CACHE.exists():
            return pd.read_parquet(GFZ_KP_CACHE)
        # Fall all the way back to the short-term NOAA Kp
        return _load_or_fetch_kp()


# ---------------------------------------------------------------------------
# merge_solar_data  –  used by features.py during training
# ---------------------------------------------------------------------------

def merge_solar_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Join Kp, SFI, and X-ray data onto the WSPR DataFrame by timestamp.

    New columns added:
      kp_index          – Kp at the time of the spot (merge_asof, nearest-past)
      kp_max_3h         – maximum Kp in the 3 hours preceding the spot
      kp_trend          – change in Kp over the preceding 3 hours (rising storm signal)
      geomagnetic_storm – 1 if kp_index >= 5 (G1 geomagnetic storm or worse)
      sfi               – daily F10.7 solar flux index
      kp_x_sfi          – interaction feature (kp_index * sfi / 100)
      xray_flux_class   – GOES X-ray class 0–4 (A/B/C/M/X)

    Historical Kp is sourced from the GFZ Potsdam definitive archive,
    which covers the full training date range. The 3-day NOAA window is
    used as a supplement for very recent rows.

    The DataFrame must have a tz-aware UTC 'timestamp' column.
    """
    df = df.copy()
    if "timestamp" not in df.columns:
        log.warning("merge_solar_data: no 'timestamp' column found; filling with defaults.")
        df["kp_index"]          = KP_DEFAULT
        df["kp_max_3h"]         = KP_DEFAULT
        df["kp_trend"]          = 0.0
        df["geomagnetic_storm"] = 0
        df["sfi"]               = SFI_DEFAULT
        df["kp_x_sfi"]          = KP_DEFAULT * SFI_DEFAULT / 100
        df["xray_flux_class"]   = XRAY_DEFAULT
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # ── Kp: merge GFZ historical + NOAA short-term into one series ────────
    gfz_df  = _load_or_fetch_gfz_kp()
    noaa_df = _load_or_fetch_kp()

    gfz_df["timestamp"]  = pd.to_datetime(gfz_df["timestamp"],  utc=True)
    noaa_df["timestamp"] = pd.to_datetime(noaa_df["timestamp"], utc=True)

    # Combine: prefer NOAA minute-resolution where it exists (recent rows),
    # use GFZ 3-hourly for everything older.
    noaa_cutoff = noaa_df["timestamp"].min() if len(noaa_df) > 0 else pd.Timestamp.now(tz="UTC")
    gfz_old     = gfz_df[gfz_df["timestamp"] < noaa_cutoff]
    kp_combined = pd.concat([gfz_old, noaa_df], ignore_index=True).sort_values("timestamp")

    df_sorted = df.sort_values("timestamp")

    # kp_index: backward nearest
    df_sorted = pd.merge_asof(
        df_sorted, kp_combined[["timestamp", "kp"]],
        on="timestamp", direction="backward",
        tolerance=pd.Timedelta("4h"),   # GFZ is 3-hourly, allow a bit of slack
    )
    df_sorted = df_sorted.rename(columns={"kp": "kp_index"})
    df_sorted["kp_index"] = df_sorted["kp_index"].fillna(KP_DEFAULT)

    # ── kp_max_3h: rolling max over preceding 3 h ─────────────────────────
    kp_ts = kp_combined.set_index("timestamp")["kp"]
    kp_rolling_max = kp_ts.rolling("3h", min_periods=1).max().rename("kp_max_3h").reset_index()
    df_sorted = pd.merge_asof(
        df_sorted, kp_rolling_max,
        on="timestamp", direction="backward",
        tolerance=pd.Timedelta("4h"),
    )
    df_sorted["kp_max_3h"] = df_sorted["kp_max_3h"].fillna(KP_DEFAULT)

    # ── kp_trend: Kp(now) – Kp(3h ago) ───────────────────────────────────
    # A rising trend (positive) is a pre-storm warning.
    kp_3h_ago = (
        kp_ts
        .reindex(kp_ts.index - pd.Timedelta("3h"), method="nearest", tolerance="3h")
        .rename("kp_3h_ago")
        .reset_index()
    )
    kp_3h_ago.columns = ["timestamp", "kp_3h_ago"]
    kp_trend_df = pd.DataFrame({
        "timestamp": kp_ts.index,
        "kp_trend":  kp_ts.values - kp_3h_ago["kp_3h_ago"].values,
    }).reset_index(drop=True)
    df_sorted = pd.merge_asof(
        df_sorted, kp_trend_df,
        on="timestamp", direction="backward",
        tolerance=pd.Timedelta("4h"),
    )
    df_sorted["kp_trend"] = df_sorted["kp_trend"].fillna(0.0)

    # ── geomagnetic_storm: Kp >= 5 (G1 storm threshold) ──────────────────
    df_sorted["geomagnetic_storm"] = (df_sorted["kp_index"] >= 5.0).astype(int)

    # ── SFI join (daily) ──────────────────────────────────────────────────
    sfi_df = _load_or_fetch_sfi()
    sfi_date = pd.to_datetime(sfi_df["date"])
    if sfi_date.dt.tz is None:
        sfi_df["date"] = sfi_date.dt.tz_localize("UTC")
    else:
        sfi_df["date"] = sfi_date.dt.tz_convert("UTC")
    sfi_df = sfi_df.sort_values("date")

    df_sorted["_date_utc"] = df_sorted["timestamp"].dt.normalize()
    df_sorted = pd.merge_asof(
        df_sorted, sfi_df.rename(columns={"date": "_date_utc"}),
        on="_date_utc", direction="backward",
        tolerance=pd.Timedelta("3d"),
    )
    df_sorted["sfi"] = df_sorted["sfi"].fillna(SFI_DEFAULT)
    df_sorted = df_sorted.drop(columns=["_date_utc"], errors="ignore")

    # ── Interaction term ──────────────────────────────────────────────────
    df_sorted["kp_x_sfi"] = df_sorted["kp_index"] * df_sorted["sfi"] / 100.0

    # ── X-ray flux (6-hour window available from NOAA) ────────────────────
    xray_df = _load_or_fetch_xray()
    xray_df["timestamp"] = pd.to_datetime(xray_df["timestamp"], utc=True)
    xray_df = xray_df.sort_values("timestamp")

    df_sorted = pd.merge_asof(
        df_sorted, xray_df[["timestamp", "xray_flux_class"]],
        on="timestamp", direction="backward",
        tolerance=pd.Timedelta("6h"),
    )
    # For historical rows outside the 6-hour X-ray window we fill with
    # the quiet A-class default; the model will learn that unknown = quiet.
    df_sorted["xray_flux_class"] = df_sorted["xray_flux_class"].fillna(XRAY_DEFAULT).astype(int)

    log.info(
        f"Solar merge done. "
        f"kp_index [{df_sorted['kp_index'].min():.1f}–{df_sorted['kp_index'].max():.1f}]  "
        f"sfi [{df_sorted['sfi'].min():.0f}–{df_sorted['sfi'].max():.0f}]  "
        f"storms: {df_sorted['geomagnetic_storm'].sum():,}  "
        f"xray_class max: {df_sorted['xray_flux_class'].max()}"
    )
    return df_sorted


# ---------------------------------------------------------------------------
# get_current_solar  –  used by app.py at inference time
# ---------------------------------------------------------------------------

def get_current_solar() -> dict[str, float]:
    """
    Return the latest solar and geomagnetic values for live inference.
    Results are cached for _LIVE_TTL seconds to avoid hammering NOAA.

    Returns dict with keys:
      kp_index, kp_max_3h, kp_trend, geomagnetic_storm,
      sfi, kp_x_sfi, xray_flux_class
    On network failure returns safe defaults.
    """
    now = time.time()
    if _live_cache.get("_expires", 0) > now:
        return {k: v for k, v in _live_cache.items() if not k.startswith("_")}

    try:
        kp_df    = fetch_kp()
        sfi_df   = fetch_sfi()
        xray_df  = fetch_xray()

        kp_latest   = float(kp_df["kp"].iloc[-1])
        kp_max_3h   = float(kp_df["kp"].tail(180).max())   # 180 × 1-min = 3 h
        kp_3h_ago   = float(kp_df["kp"].iloc[max(0, len(kp_df) - 180)])
        kp_trend    = float(kp_latest - kp_3h_ago)
        sfi_latest  = float(sfi_df["sfi"].iloc[-1])
        kp_x_sfi    = kp_latest * sfi_latest / 100.0
        xray_class  = int(xray_df["xray_flux_class"].iloc[-1]) if len(xray_df) > 0 else XRAY_DEFAULT
        storm_flag  = int(kp_latest >= 5.0)

        result = {
            "kp_index":          kp_latest,
            "kp_max_3h":         kp_max_3h,
            "kp_trend":          kp_trend,
            "geomagnetic_storm": storm_flag,
            "sfi":               sfi_latest,
            "kp_x_sfi":         kp_x_sfi,
            "xray_flux_class":   xray_class,
        }
        _live_cache.update(result)
        _live_cache["_expires"] = now + _LIVE_TTL
        log.info(
            f"Live solar: kp={kp_latest:.1f}  kp_trend={kp_trend:+.1f}  "
            f"storm={storm_flag}  sfi={sfi_latest:.0f}  xray_class={xray_class}"
        )
        return result

    except Exception as e:
        log.warning(f"Live solar fetch failed ({e}); using defaults.")
        return {
            "kp_index":          KP_DEFAULT,
            "kp_max_3h":         KP_DEFAULT,
            "kp_trend":          0.0,
            "geomagnetic_storm": 0,
            "sfi":               SFI_DEFAULT,
            "kp_x_sfi":          KP_DEFAULT * SFI_DEFAULT / 100.0,
            "xray_flux_class":   XRAY_DEFAULT,
        }


# ---------------------------------------------------------------------------
# get_current_alerts  –  used by app.py /solar_status endpoint
# ---------------------------------------------------------------------------

def get_current_alerts() -> list[dict]:
    """
    Return the latest NOAA SWPC alert/watch/warning messages.
    Cached for _ALERTS_TTL seconds.
    """
    now = time.time()
    if _alerts_cache.get("_expires", 0) > now:
        return _alerts_cache.get("alerts", [])

    try:
        raw = fetch_alerts()
        # Normalise: keep only the fields we need
        alerts = []
        for entry in raw:
            msg_type = entry.get("product_id", "")
            issue    = entry.get("issue_datetime", "")
            message  = entry.get("message", "")
            if not message:
                continue
            # Derive a clean summary from the first non-empty line of the message
            summary  = next((l.strip() for l in message.splitlines() if l.strip()), message[:120])
            alerts.append({
                "product_id":    msg_type,
                "issue_datetime": issue,
                "summary":       summary,
                "message":       message,
            })
        _alerts_cache["alerts"]   = alerts
        _alerts_cache["_expires"] = now + _ALERTS_TTL
        log.info(f"SWPC alerts fetched: {len(alerts)} entries")
        return alerts
    except Exception as e:
        log.warning(f"SWPC alerts fetch failed ({e}); returning empty list.")
        return _alerts_cache.get("alerts", [])
