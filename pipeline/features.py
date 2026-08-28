"""
Step 4 – Feature Engineering
------------------------------
Computes per-location, per-time-window features from the labeled
WSPR dataset.

Feature groups:
  SNR features      – rolling mean/min/max/std, snr_diff, snr_slope
  Time-gap features – time_gap_min, rolling max/min/mean of gaps, gap_acceleration
  Spot density      – spots per 10-min, 30-min; spot_rate_trend
  Outage recency    – mins_since_last_outage
  Frequency/band    – frequency_hz, band_enc
  Temporal          – hour_of_day, day_of_week, solar_zenith_angle, is_daytime
  Solar/geomagnetic – kp_index, kp_max_3h, kp_trend, geomagnetic_storm,
                      sfi, kp_x_sfi, xray_flux_class

Output columns appended to input DataFrame; all NaN rows (due to
rolling windows at the start of each grid's series) are filled with
median values so the model always receives a complete feature vector.
"""

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
INPUT_FILE    = PROCESSED_DIR / "wspr_labeled.parquet"
OUTPUT_FILE   = PROCESSED_DIR / "wspr_features.parquet"

ROLLING_N_SPOTS   = 5          # rolling window over N consecutive spots
SNR_FILL_DEFAULT  = -20.0      # fallback when SNR is entirely missing

FEATURE_COLS = [
    # SNR features
    "snr_rolling_mean", "snr_rolling_min", "snr_rolling_max", "snr_rolling_std",
    "snr_diff",
    # SNR trend: slope of SNR over last N spots (leading indicator of degradation)
    "snr_slope",
    # Time-gap features
    "time_gap_min", "gap_rolling_max", "gap_rolling_min", "gap_rolling_mean",
    # Gap acceleration: are gaps getting longer? (diff of rolling mean gap)
    "gap_acceleration",
    # Spot density
    "spots_per_10min", "spots_per_30min",
    # Spot rate trend: recent 30-min density minus prior 30-min density
    "spot_rate_trend",
    # Outage recency: minutes since last outage at this grid
    "mins_since_last_outage",
    "frequency_hz",
    "band_enc",
    "hour_of_day", "day_of_week",
    # Solar zenith angle (degrees): replaces the blunt is_daytime UTC window.
    # Computed from the grid's lat/lon, captures D-layer absorption precisely.
    "solar_zenith_angle",
    # Kept for backwards compat but now derived from solar_zenith_angle
    "is_daytime",
    # Solar / geomagnetic features
    "kp_index", "kp_max_3h", "kp_trend", "geomagnetic_storm",
    "sfi", "kp_x_sfi",
    # GOES X-ray flux class (A=0, B=1, C=2, M=3, X=4)
    # Captures sudden ionospheric disturbances from solar flares
    "xray_flux_class",
]

TARGET_COL = "future_outage_label"


# ---------------------------------------------------------------------------
# Solar zenith angle helper
# ---------------------------------------------------------------------------

def _solar_zenith_angle(ts: pd.Timestamp, lat: float, lon: float) -> float:
    """
    Compute the solar zenith angle (degrees) for a given UTC timestamp and
    geographic location using the simplified NOAA solar position algorithm.

    0°  = sun directly overhead (noon at equator on equinox)
    90° = sun on the horizon (sunrise / sunset)
    >90° = night

    Accuracy is ~0.01° — more than sufficient for HF propagation modelling.
    Returns 90.0 (horizon) for invalid/missing coordinates.
    """
    if lat is None or lon is None or math.isnan(lat) or math.isnan(lon):
        return 90.0

    # Day of year
    doy     = ts.day_of_year
    # Fractional hour in UTC
    hr_utc  = ts.hour + ts.minute / 60.0 + ts.second / 3600.0

    # Solar declination (degrees) — Spencer's formula
    B       = math.radians((360 / 365) * (doy - 1))
    decl    = (
        0.006918
        - 0.399912 * math.cos(B)
        + 0.070257 * math.sin(B)
        - 0.006758 * math.cos(2 * B)
        + 0.000907 * math.sin(2 * B)
        - 0.002697 * math.cos(3 * B)
        + 0.00148  * math.sin(3 * B)
    )  # radians

    # Equation of time (minutes)
    eot = 229.18 * (
        0.000075
        + 0.001868 * math.cos(B)
        - 0.032077 * math.sin(B)
        - 0.014615 * math.cos(2 * B)
        - 0.04089  * math.sin(2 * B)
    )

    # True solar time (minutes from midnight)
    tst = hr_utc * 60 + eot + 4 * lon

    # Hour angle (degrees)
    ha = (tst / 4) - 180

    # Solar zenith angle
    lat_r   = math.radians(lat)
    ha_r    = math.radians(ha)
    cos_sza = (
        math.sin(lat_r) * math.sin(decl)
        + math.cos(lat_r) * math.cos(decl) * math.cos(ha_r)
    )
    cos_sza = max(-1.0, min(1.0, cos_sza))
    return math.degrees(math.acos(cos_sza))


def _solar_zenith_series(df_grid: pd.DataFrame) -> pd.Series:
    """
    Vectorised solar zenith angle for a single-grid DataFrame.
    Uses the grid's representative lat/lon (first non-null values).
    """
    lat = df_grid["tx_lat"].dropna().iloc[0]  if ("tx_lat" in df_grid.columns and df_grid["tx_lat"].notna().any()) else None
    lon = df_grid["tx_lon"].dropna().iloc[0]  if ("tx_lon" in df_grid.columns and df_grid["tx_lon"].notna().any()) else None

    if lat is None or lon is None:
        return pd.Series(90.0, index=df_grid.index)

    return df_grid["timestamp"].apply(lambda ts: _solar_zenith_angle(ts, lat, lon))


# ---------------------------------------------------------------------------
# Per-grid feature computation
# ---------------------------------------------------------------------------
def _features_single_grid(df_grid: pd.DataFrame) -> pd.DataFrame:
    df_grid = df_grid.sort_values("timestamp").copy().reset_index(drop=True)

    # ── SNR features ────────────────────────────────────────────────────
    if "snr" in df_grid.columns:
        snr = df_grid["snr"].fillna(SNR_FILL_DEFAULT)
        df_grid["snr_rolling_mean"] = snr.rolling(ROLLING_N_SPOTS, min_periods=1).mean()
        df_grid["snr_rolling_min"]  = snr.rolling(ROLLING_N_SPOTS, min_periods=1).min()
        df_grid["snr_rolling_max"]  = snr.rolling(ROLLING_N_SPOTS, min_periods=1).max()
        df_grid["snr_rolling_std"]  = snr.rolling(ROLLING_N_SPOTS, min_periods=1).std().fillna(0)
        df_grid["snr_diff"]         = snr.diff().fillna(0)
        # SNR slope: linear trend over last ROLLING_N_SPOTS spots
        # Positive = improving, negative = degrading (pre-outage signal)
        def _slope(s):
            if len(s) < 2:
                return 0.0
            x = np.arange(len(s), dtype=float)
            x -= x.mean()
            denom = (x * x).sum()
            return float((x * (s.values - s.values.mean())).sum() / denom) if denom else 0.0
        df_grid["snr_slope"] = snr.rolling(ROLLING_N_SPOTS, min_periods=2).apply(_slope, raw=False).fillna(0)
    else:
        for col in ["snr_rolling_mean", "snr_rolling_min", "snr_rolling_max",
                    "snr_rolling_std", "snr_diff", "snr_slope"]:
            df_grid[col] = SNR_FILL_DEFAULT

    # ── Time-gap features ────────────────────────────────────────────────
    if "time_gap_min" not in df_grid.columns:
        df_grid["time_gap_min"] = (
            df_grid["timestamp"].diff().dt.total_seconds() / 60
        ).fillna(0)
    else:
        df_grid["time_gap_min"] = df_grid["time_gap_min"].fillna(0)

    gap = df_grid["time_gap_min"]
    df_grid["gap_rolling_max"]  = gap.rolling(ROLLING_N_SPOTS, min_periods=1).max()
    df_grid["gap_rolling_min"]  = gap.rolling(ROLLING_N_SPOTS, min_periods=1).min()
    df_grid["gap_rolling_mean"] = gap.rolling(ROLLING_N_SPOTS, min_periods=1).mean()
    # Gap acceleration: diff of rolling mean — positive = gaps widening (pre-outage)
    df_grid["gap_acceleration"] = df_grid["gap_rolling_mean"].diff().fillna(0)

    # ── Spot-density features  (time-indexed resampling) ─────────────────
    ts_idx = df_grid.set_index("timestamp")
    spot_1min = ts_idx.resample("1min").size().rename("_cnt")

    rolling_10 = spot_1min.rolling("10min", min_periods=1).sum()
    rolling_30 = spot_1min.rolling("30min", min_periods=1).sum()
    rolling_60 = spot_1min.rolling("60min", min_periods=1).sum()

    # Map back to original rows (merge_asof on nearest minute)
    df_grid = df_grid.sort_values("timestamp")
    df_grid["_minute"] = df_grid["timestamp"].dt.floor("1min")
    density_df = pd.DataFrame({
        "minute":          spot_1min.index,
        "spots_per_10min": rolling_10.values,
        "spots_per_30min": rolling_30.values,
        "spots_per_60min": rolling_60.values,
    })
    df_grid = pd.merge_asof(
        df_grid, density_df, left_on="_minute", right_on="minute",
        direction="backward"
    ).drop(columns=["_minute", "minute"], errors="ignore")

    # Spot rate trend: last-30min density minus prior-30min density
    # Negative = spot rate falling = pre-outage signal
    df_grid["spot_rate_trend"] = (
        df_grid["spots_per_30min"] -
        (df_grid["spots_per_60min"] - df_grid["spots_per_30min"])
    ).fillna(0)

    # ── Frequency ────────────────────────────────────────────────────────
    if "frequency_hz" in df_grid.columns:
        df_grid["frequency_hz"] = pd.to_numeric(df_grid["frequency_hz"], errors="coerce").fillna(0)
    else:
        df_grid["frequency_hz"] = 0.0

    # ── Outage recency ───────────────────────────────────────────────────
    # Minutes since the most recent outage_label=1 at this grid.
    if "outage_label" in df_grid.columns:
        outage_times = df_grid.loc[df_grid["outage_label"] == 1, "timestamp"]
        if len(outage_times) > 0:
            ts_arr  = df_grid["timestamp"].values.astype("datetime64[ns]")
            out_arr = outage_times.values.astype("datetime64[ns]")
            pos = np.searchsorted(out_arr, ts_arr, side="right") - 1
            valid = pos >= 0
            mins_since = np.full(len(ts_arr), np.nan)
            mins_since[valid] = (
                (ts_arr[valid].astype(np.int64) - out_arr[pos[valid]].astype(np.int64))
                / 1e9 / 60
            )
            df_grid["mins_since_last_outage"] = mins_since
        else:
            df_grid["mins_since_last_outage"] = np.nan
    else:
        df_grid["mins_since_last_outage"] = np.nan

    # ── Temporal features ────────────────────────────────────────────────
    df_grid["hour_of_day"] = df_grid["timestamp"].dt.hour
    df_grid["day_of_week"] = df_grid["timestamp"].dt.dayofweek

    # Solar zenith angle: precise day/night boundary per location
    df_grid["solar_zenith_angle"] = _solar_zenith_series(df_grid)
    # is_daytime: 1 when sun is above the horizon (SZA < 90°)
    df_grid["is_daytime"] = (df_grid["solar_zenith_angle"] < 90.0).astype(int)

    return df_grid


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering per grid square, encode band."""
    results = []
    grids = df["tx_grid"].dropna().unique() if "tx_grid" in df.columns else [None]
    log.info(f"Engineering features for {len(grids)} grid squares …")

    for grid in grids:
        subset = (df[df["tx_grid"] == grid].copy() if grid is not None else df.copy())
        results.append(_features_single_grid(subset))

    combined = pd.concat(results, ignore_index=True).sort_values("timestamp")

    # ── Band label encoding ───────────────────────────────────────────────
    if "band" in combined.columns:
        le = LabelEncoder()
        combined["band_enc"] = le.fit_transform(combined["band"].fillna("unknown"))
        band_classes = list(le.classes_)
    else:
        combined["band_enc"] = 0
        band_classes = []

    # ── Ensure solar columns have defaults if merge_solar_data was skipped ─
    solar_defaults = {
        "kp_index":          2.0,
        "kp_max_3h":         2.0,
        "kp_trend":          0.0,
        "geomagnetic_storm": 0,
        "sfi":               120.0,
        "kp_x_sfi":          2.4,
        "xray_flux_class":   0,
    }
    for col, default in solar_defaults.items():
        if col not in combined.columns:
            combined[col] = default

    # ── Fill any residual NaNs in feature cols ────────────────────────────
    # mins_since_last_outage: NaN means no prior outage recorded → use a large
    # sentinel (1 week) so the model learns "no recent outage" as a distinct signal.
    if "mins_since_last_outage" in combined.columns:
        combined["mins_since_last_outage"] = combined["mins_since_last_outage"].fillna(60 * 24 * 7)

    for col in FEATURE_COLS:
        if col in combined.columns and col != "mins_since_last_outage":
            combined[col] = combined[col].fillna(combined[col].median())

    log.info(f"Feature engineering complete. Shape: {combined.shape}")
    return combined, band_classes


def run(input_file: Path = INPUT_FILE, output_file: Path = OUTPUT_FILE):
    from pipeline.solar import merge_solar_data
    df = pd.read_parquet(input_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    log.info("Merging solar/geomagnetic data …")
    df = merge_solar_data(df)
    featured, band_classes = engineer_features(df)
    featured.to_parquet(output_file, index=False)
    log.info(f"Features dataset saved → {output_file}")

    import json
    meta_path = output_file.parent / "feature_meta.json"
    meta = {
        "feature_cols":    FEATURE_COLS,
        "target_col":      TARGET_COL,
        "band_classes":    [str(c) for c in band_classes],
        "rolling_n_spots": int(ROLLING_N_SPOTS),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"Feature meta saved → {meta_path}")
    return featured


if __name__ == "__main__":
    run()
