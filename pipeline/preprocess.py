"""
Shared preprocessing / feature computation used by both the training
pipeline and the live prediction service.

This ensures that training and inference always apply identical
transformations.

Solar/geomagnetic values are accepted as optional kwargs by
build_feature_vector and passed through to the feature dict.
The caller (app.py) is responsible for fetching live values via
pipeline.solar.get_current_solar().
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
FEATURE_META  = PROCESSED_DIR / "feature_meta.json"

# Solar fallbacks (quiet-day / moderate-activity defaults)
KP_DEFAULT    = 2.0
SFI_DEFAULT   = 120.0
XRAY_DEFAULT  = 0     # A-class (quiet)

# Defaults used when no meta file is available
_DEFAULT_FEATURE_COLS = [
    "snr_rolling_mean", "snr_rolling_min", "snr_rolling_max", "snr_rolling_std",
    "snr_diff", "snr_slope",
    "time_gap_min", "gap_rolling_max", "gap_rolling_min", "gap_rolling_mean",
    "gap_acceleration",
    "spots_per_10min", "spots_per_30min",
    "spot_rate_trend",
    "mins_since_last_outage",
    "frequency_hz",
    "band_enc",
    "hour_of_day", "day_of_week", "solar_zenith_angle", "is_daytime",
    "kp_index", "kp_max_3h", "kp_trend", "geomagnetic_storm",
    "sfi", "kp_x_sfi",
    "xray_flux_class",
]

# Sentinel for mins_since_last_outage when no prior outage is known (1 week)
MINS_SINCE_OUTAGE_DEFAULT = 60 * 24 * 7

SNR_FILL_DEFAULT  = -20.0
ROLLING_N_SPOTS   = 5


def load_feature_meta() -> dict:
    try:
        with open(FEATURE_META) as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "feature_cols": _DEFAULT_FEATURE_COLS,
            "band_classes": [],
            "rolling_n_spots": ROLLING_N_SPOTS,
        }


def band_encode(band_str: Optional[str], band_classes: list) -> int:
    """Return integer encoding of a band string, matching training LabelEncoder."""
    if not band_classes:
        return 0
    try:
        return band_classes.index(band_str) if band_str in band_classes else 0
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# Solar zenith angle  (mirrors features.py — kept in sync manually)
# ---------------------------------------------------------------------------

def _solar_zenith_angle(ts: pd.Timestamp, lat: float, lon: float) -> float:
    """
    Compute the solar zenith angle (degrees) for a given UTC timestamp and
    geographic location using the simplified NOAA solar position algorithm.

    0°  = sun directly overhead
    90° = horizon (sunrise/sunset)
    >90° = night

    Returns 90.0 for invalid/missing coordinates.
    """
    if lat is None or lon is None:
        return 90.0
    try:
        if math.isnan(lat) or math.isnan(lon):
            return 90.0
    except (TypeError, ValueError):
        return 90.0

    doy    = ts.day_of_year
    hr_utc = ts.hour + ts.minute / 60.0 + ts.second / 3600.0

    B      = math.radians((360 / 365) * (doy - 1))
    decl   = (
        0.006918
        - 0.399912 * math.cos(B)
        + 0.070257 * math.sin(B)
        - 0.006758 * math.cos(2 * B)
        + 0.000907 * math.sin(2 * B)
        - 0.002697 * math.cos(3 * B)
        + 0.00148  * math.sin(3 * B)
    )  # radians

    eot = 229.18 * (
        0.000075
        + 0.001868 * math.cos(B)
        - 0.032077 * math.sin(B)
        - 0.014615 * math.cos(2 * B)
        - 0.04089  * math.sin(2 * B)
    )

    tst    = hr_utc * 60 + eot + 4 * lon
    ha     = (tst / 4) - 180
    lat_r  = math.radians(lat)
    ha_r   = math.radians(ha)
    cos_sza = (
        math.sin(lat_r) * math.sin(decl)
        + math.cos(lat_r) * math.cos(decl) * math.cos(ha_r)
    )
    cos_sza = max(-1.0, min(1.0, cos_sza))
    return math.degrees(math.acos(cos_sza))


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------

def build_feature_vector(
    spots: pd.DataFrame,
    current_time: pd.Timestamp,
    band_classes: list,
    # Location — needed for solar zenith angle
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    # Solar / geomagnetic
    kp_index: float = KP_DEFAULT,
    kp_max_3h: float = KP_DEFAULT,
    kp_trend: float = 0.0,
    geomagnetic_storm: int = 0,
    sfi: float = SFI_DEFAULT,
    kp_x_sfi: Optional[float] = None,
    xray_flux_class: int = XRAY_DEFAULT,
) -> dict:
    """
    Given a DataFrame of recent WSPR spots (columns: timestamp, snr,
    frequency_hz, band) at a single location, build the feature dict
    expected by the Random Forest model.

    spots should already be sorted by timestamp, ascending, and filtered
    to the relevant location.

    Solar/geomagnetic kwargs:
      kp_index          – instantaneous Kp (0–9)
      kp_max_3h         – max Kp in preceding 3 hours
      kp_trend          – Kp(now) - Kp(3h ago); positive = rising storm
      geomagnetic_storm – 1 if kp_index >= 5
      sfi               – F10.7 solar flux index
      kp_x_sfi          – interaction term; computed if None
      xray_flux_class   – GOES X-ray class 0 (A) to 4 (X)

    Location kwargs:
      lat, lon          – decimal degrees; used for solar zenith angle
    """
    if kp_x_sfi is None:
        kp_x_sfi = kp_index * sfi / 100.0

    n = ROLLING_N_SPOTS

    # ── SNR features ─────────────────────────────────────────────────────
    if "snr" in spots.columns and not spots["snr"].isna().all():
        snr = spots["snr"].fillna(SNR_FILL_DEFAULT).tail(n)
        snr_mean = float(snr.mean())
        snr_min  = float(snr.min())
        snr_max  = float(snr.max())
        snr_std  = float(snr.std()) if len(snr) > 1 else 0.0
        snr_diff = float(snr.iloc[-1] - snr.iloc[-2]) if len(snr) >= 2 else 0.0
        if len(snr) >= 2:
            x = np.arange(len(snr), dtype=float)
            x -= x.mean()
            denom = float((x * x).sum())
            snr_slope = float((x * (snr.values - snr.values.mean())).sum() / denom) if denom else 0.0
        else:
            snr_slope = 0.0
    else:
        snr_mean = snr_min = snr_max = SNR_FILL_DEFAULT
        snr_std  = snr_diff = snr_slope = 0.0

    # ── Time-gap features ─────────────────────────────────────────────────
    if len(spots) >= 2:
        gaps = spots["timestamp"].diff().dt.total_seconds().div(60).dropna()
        last_gaps = gaps.tail(n)
        time_gap  = float(gaps.iloc[-1]) if len(gaps) > 0 else 0.0
        gap_max   = float(last_gaps.max())
        gap_min   = float(last_gaps.min())
        gap_mean  = float(last_gaps.mean())
        gap_accel = float(last_gaps.diff().iloc[-1]) if len(last_gaps) >= 2 else 0.0
    else:
        time_gap = gap_max = gap_min = gap_mean = gap_accel = 0.0

    # ── Spot-density features ─────────────────────────────────────────────
    window_10 = spots[spots["timestamp"] >= current_time - pd.Timedelta(minutes=10)]
    window_30 = spots[spots["timestamp"] >= current_time - pd.Timedelta(minutes=30)]
    window_60 = spots[spots["timestamp"] >= current_time - pd.Timedelta(minutes=60)]
    spots_10  = float(len(window_10))
    spots_30  = float(len(window_30))
    spots_60  = float(len(window_60))
    prior_30  = spots_60 - spots_30
    spot_rate_trend = float(spots_30 - prior_30)

    # ── Outage recency ────────────────────────────────────────────────────
    last_outage_time: Optional[pd.Timestamp] = getattr(spots, "_last_outage_time", None)
    if last_outage_time is not None:
        mins_since = float((current_time - last_outage_time).total_seconds() / 60)
        mins_since = max(mins_since, 0.0)
    else:
        mins_since = float(MINS_SINCE_OUTAGE_DEFAULT)

    # ── Frequency / band ─────────────────────────────────────────────────
    freq_hz  = float(spots["frequency_hz"].iloc[-1]) if "frequency_hz" in spots.columns and len(spots) > 0 else 0.0
    band_str = spots["band"].iloc[-1] if "band" in spots.columns and len(spots) > 0 else None
    band_enc = band_encode(band_str, band_classes)

    # ── Temporal ──────────────────────────────────────────────────────────
    hour = current_time.hour
    dow  = current_time.dayofweek

    # Solar zenith angle: use provided lat/lon or fall back to spots columns
    _lat = lat
    _lon = lon
    if _lat is None and "tx_lat" in spots.columns and len(spots) > 0:
        _lat = spots["tx_lat"].dropna().iloc[0] if spots["tx_lat"].notna().any() else None
    if _lon is None and "tx_lon" in spots.columns and len(spots) > 0:
        _lon = spots["tx_lon"].dropna().iloc[0] if spots["tx_lon"].notna().any() else None

    sza        = _solar_zenith_angle(current_time, _lat, _lon)
    is_daytime = int(sza < 90.0)

    return {
        "snr_rolling_mean":       snr_mean,
        "snr_rolling_min":        snr_min,
        "snr_rolling_max":        snr_max,
        "snr_rolling_std":        snr_std,
        "snr_diff":               snr_diff,
        "snr_slope":              snr_slope,
        "time_gap_min":           time_gap,
        "gap_rolling_max":        gap_max,
        "gap_rolling_min":        gap_min,
        "gap_rolling_mean":       gap_mean,
        "gap_acceleration":       gap_accel,
        "spots_per_10min":        spots_10,
        "spots_per_30min":        spots_30,
        "spot_rate_trend":        spot_rate_trend,
        "mins_since_last_outage": mins_since,
        "frequency_hz":           freq_hz,
        "band_enc":               band_enc,
        "hour_of_day":            hour,
        "day_of_week":            dow,
        "solar_zenith_angle":     sza,
        "is_daytime":             is_daytime,
        # Solar / geomagnetic
        "kp_index":               kp_index,
        "kp_max_3h":              kp_max_3h,
        "kp_trend":               kp_trend,
        "geomagnetic_storm":      geomagnetic_storm,
        "sfi":                    sfi,
        "kp_x_sfi":               kp_x_sfi,
        "xray_flux_class":        xray_flux_class,
    }
