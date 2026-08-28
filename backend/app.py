"""
Prediction Service  –  FastAPI backend
POST /predict_outage
GET  /health
GET  /stations          (list of known grid squares with lat/lon)
GET  /heatmap_data      (batch probabilities for all known stations)
"""

from __future__ import annotations

import json
import logging
import threading
from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from cachetools import TTLCache
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).resolve().parents[1] / "logs" / "predictions.log"),
    ],
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parents[1]
MODELS_DIR    = BASE_DIR / "models"
DATA_DIR      = BASE_DIR / "data" / "processed"
FRONTEND_DIR  = BASE_DIR / "frontend"

MODEL_FILE    = MODELS_DIR / "rf_outage_model.pkl"
SCHEMA_FILE   = MODELS_DIR / "model_schema.json"
FEATURES_FILE = DATA_DIR   / "wspr_features.parquet"

# Per-band model helpers — mirrors train.py naming
def _band_model_file(band_enc: int) -> Path:
    return MODELS_DIR / f"rf_outage_model_band_{band_enc}.pkl"

def _band_schema_file(band_enc: int) -> Path:
    return MODELS_DIR / f"model_schema_band_{band_enc}.json"

HORIZON_MINUTES      = 60
FORWARD_STEPS        = 6            # number of future windows to evaluate
STEP_MINUTES         = HORIZON_MINUTES // FORWARD_STEPS
RECENT_WINDOW_HOURS  = 3
CACHE_TTL_SECONDS    = 300

# How many of the most-recent spots per grid to keep in the compact cache.
# 50 is more than enough for the rolling-5-spot and 30/60-min density features.
SPOTS_PER_GRID       = 50

# Columns needed for inference (skip raw text cols to save memory)
_INFER_COLS = [
    "tx_grid", "tx_lat", "tx_lon", "timestamp",
    "snr", "frequency_hz", "band", "band_enc",
]

# ── App ───────────────────────────────────────────────────────────────────
app = FastAPI(title="WSPR Outage Predictor", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
if (FRONTEND_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")


# ── Prediction cache  (location+time → result) ───────────────────────────
_prediction_cache: TTLCache = TTLCache(maxsize=1024, ttl=CACHE_TTL_SECONDS)


# ── Load model (lazy, cached) ─────────────────────────────────────────────
@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_FILE.exists():
        raise RuntimeError(f"Model not found: {MODEL_FILE}. Run pipeline/train.py first.")
    rf = joblib.load(MODEL_FILE)
    log.info(f"Model loaded from {MODEL_FILE}")
    return rf


@lru_cache(maxsize=1)
def _load_schema() -> dict:
    with open(SCHEMA_FILE) as f:
        return json.load(f)


@lru_cache(maxsize=8)
def _load_band_model(band_enc: int):
    """Load a per-band RF model; returns None if not available."""
    path = _band_model_file(band_enc)
    if not path.exists():
        return None
    try:
        m = joblib.load(path)
        log.info(f"Band model loaded: band_enc={band_enc} from {path.name}")
        return m
    except Exception as e:
        log.warning(f"Could not load band model {path.name}: {e}")
        return None


@lru_cache(maxsize=8)
def _load_band_schema(band_enc: int) -> Optional[dict]:
    """Load per-band schema; returns None if not available."""
    path = _band_schema_file(band_enc)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


# ── Compact per-grid spot cache ───────────────────────────────────────────
# Built once at startup by streaming through the parquet file in chunks.
# Structure: { grid_str -> pd.DataFrame (≤ SPOTS_PER_GRID rows, sorted by ts) }
_grid_spot_cache: dict[str, pd.DataFrame] = {}
_grid_meta: dict[str, tuple[float, float]] = {}   # grid -> (lat, lon)
_data_end_ts: Optional[pd.Timestamp] = None
_cache_ready = False
_cache_lock  = threading.Lock()


def _build_spot_cache() -> None:
    """
    Stream through wspr_features.parquet one row-group at a time, keeping
    only the most-recent SPOTS_PER_GRID rows per grid.  Uses ~1-5% of the
    memory that a full pd.read_parquet() would require.
    Called once in a background thread at startup.
    """
    global _grid_spot_cache, _grid_meta, _data_end_ts, _cache_ready

    if not FEATURES_FILE.exists():
        log.warning("Features file not found; spot cache will be empty.")
        _cache_ready = True
        return

    log.info("Building per-grid spot cache from parquet (streaming)…")

    # Figure out which columns are actually in the file
    pf       = pq.ParquetFile(str(FEATURES_FILE))
    all_cols = pf.schema_arrow.names
    read_cols = [c for c in _INFER_COLS if c in all_cols]

    # We accumulate per-grid buffers, then trim at the end.
    # Use a plain dict of lists for speed; convert to DataFrame once.
    buffers: dict[str, list] = {}
    global_max_ts: Optional[pd.Timestamp] = None

    for rg_idx in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(rg_idx, columns=read_cols)
        chunk = table.to_pandas()
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], utc=True)

        # Track global dataset end time
        rg_max = chunk["timestamp"].max()
        if global_max_ts is None or rg_max > global_max_ts:
            global_max_ts = rg_max

        # Accumulate rows per grid
        for grid, grp in chunk.groupby("tx_grid", sort=False):
            if grid not in buffers:
                buffers[grid] = []
            buffers[grid].append(grp)

        if (rg_idx + 1) % 5 == 0:
            log.info(f"  …processed {rg_idx + 1}/{pf.metadata.num_row_groups} row groups, "
                     f"{len(buffers)} grids so far")

    # Build final per-grid DataFrames: keep only the last SPOTS_PER_GRID rows
    spot_cache: dict[str, pd.DataFrame] = {}
    grid_meta:  dict[str, tuple[float, float]] = {}

    for grid, parts in buffers.items():
        df = pd.concat(parts, ignore_index=True).sort_values("timestamp")
        df = df.tail(SPOTS_PER_GRID).reset_index(drop=True)
        spot_cache[grid] = df
        # Extract lat/lon from the last row with valid coords
        if "tx_lat" in df.columns and "tx_lon" in df.columns:
            valid = df[df["tx_lat"].notna() & df["tx_lon"].notna()]
            if not valid.empty:
                grid_meta[grid] = (float(valid["tx_lat"].iloc[-1]),
                                   float(valid["tx_lon"].iloc[-1]))

    with _cache_lock:
        _grid_spot_cache = spot_cache
        _grid_meta       = grid_meta
        _data_end_ts     = global_max_ts
        _cache_ready     = True

    log.info(f"Spot cache ready: {len(spot_cache)} grids, "
             f"dataset end={global_max_ts.isoformat() if global_max_ts else 'unknown'}")


def _ensure_cache() -> bool:
    """Return True if the spot cache is ready."""
    return _cache_ready


# Kick off background cache build immediately at import time
threading.Thread(target=_build_spot_cache, daemon=True, name="spot-cache-builder").start()


def _load_grid_index() -> pd.DataFrame:
    """
    Return a DataFrame: one row per unique tx_grid with tx_lat / tx_lon.
    Derived from the in-memory spot cache — no parquet read needed.
    """
    if not _grid_meta:
        return pd.DataFrame(columns=["tx_grid", "tx_lat", "tx_lon"])
    rows = [{"tx_grid": g, "tx_lat": lat, "tx_lon": lon}
            for g, (lat, lon) in _grid_meta.items()]
    return pd.DataFrame(rows)


def _data_end_time() -> pd.Timestamp:
    """Latest timestamp in the features dataset (from the compact cache)."""
    if _data_end_ts is not None:
        return _data_end_ts
    return pd.Timestamp.now(tz="UTC")


def _effective_time(requested: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    """
    Return the time to use for spot lookups.
    If the requested time (or now) is past the dataset's last timestamp,
    fall back to the dataset end so the 3-hour lookback always finds data.
    """
    t = requested if requested is not None else pd.Timestamp.now(tz="UTC")
    end = _data_end_time()
    if t > end:
        return end
    return t


# ── Schemas ───────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    latitude:    Optional[float] = Field(None, description="Decimal latitude")
    longitude:   Optional[float] = Field(None, description="Decimal longitude")
    grid_square: Optional[str]   = Field(None, description="Maidenhead grid (e.g. FN42)")
    current_time: Optional[str]  = Field(None, description="ISO-8601 UTC time; defaults to now")


class LocationInfo(BaseModel):
    latitude:    Optional[float]
    longitude:   Optional[float]
    grid_square: Optional[str]


class PredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    location:          LocationInfo
    next_outage_start: Optional[str]
    next_outage_end:   Optional[str]
    probability:       float
    risk_level:        str              # low | medium | high
    confidence_score:  float
    prediction_unavailable: bool = False
    unavailable_reason: Optional[str] = None
    model_version:     str = "1.0"


# ── Helpers ───────────────────────────────────────────────────────────────
def _risk_level(prob: float) -> str:
    if prob < 0.33:
        return "low"
    if prob < 0.66:
        return "medium"
    return "high"


def _nearest_grid(lat: float, lon: float, known_grids: pd.DataFrame) -> str:
    """Return the grid square in known_grids closest to (lat, lon)."""
    if known_grids.empty:
        return ""
    lat_col = "tx_lat" if "tx_lat" in known_grids.columns else "latitude"
    lon_col = "tx_lon" if "tx_lon" in known_grids.columns else "longitude"
    dists = np.sqrt((known_grids[lat_col] - lat) ** 2 + (known_grids[lon_col] - lon) ** 2)
    idx = dists.idxmin()
    grid_col = "tx_grid" if "tx_grid" in known_grids.columns else known_grids.columns[0]
    return known_grids.loc[idx, grid_col]


def _recent_spots(grid: str, current_time: pd.Timestamp,
                   hours: int = RECENT_WINDOW_HOURS) -> pd.DataFrame:
    """
    Return recent rows for a grid from the compact in-memory cache.
    O(1) lookup — no full-DataFrame scan.
    If no rows fall in the requested window, progressively widens up to
    the full cached history (always ≤ SPOTS_PER_GRID rows).
    """
    grid_df = _grid_spot_cache.get(grid)
    if grid_df is None or grid_df.empty:
        return pd.DataFrame()
    # Try progressively wider windows
    for window_hours in [hours, 6, 12, 24, 48, 72]:
        cutoff = current_time - pd.Timedelta(hours=window_hours)
        mask   = (grid_df["timestamp"] >= cutoff) & (grid_df["timestamp"] <= current_time)
        result = grid_df[mask]
        if not result.empty:
            return result
    # Last resort: return all cached rows for this grid regardless of time
    return grid_df


def _build_feature_row(spots: pd.DataFrame, current_time: pd.Timestamp,
                        band_classes: list, feature_cols: list,
                        solar: dict | None = None,
                        lat: Optional[float] = None,
                        lon: Optional[float] = None) -> np.ndarray:
    """Build a single feature vector using shared preprocess logic."""
    from pipeline.preprocess import build_feature_vector
    solar = solar or {}
    fvec = build_feature_vector(
        spots, current_time, band_classes,
        lat=lat,
        lon=lon,
        kp_index=solar.get("kp_index", 2.0),
        kp_max_3h=solar.get("kp_max_3h", 2.0),
        kp_trend=solar.get("kp_trend", 0.0),
        geomagnetic_storm=int(solar.get("geomagnetic_storm", 0)),
        sfi=solar.get("sfi", 120.0),
        kp_x_sfi=solar.get("kp_x_sfi"),
        xray_flux_class=int(solar.get("xray_flux_class", 0)),
    )
    row = [fvec.get(col, 0.0) for col in feature_cols]
    return np.array(row, dtype=float).reshape(1, -1)


def _predict_outage(rf, schema: dict, spots: pd.DataFrame,
                     current_time: pd.Timestamp,
                     solar: dict | None = None,
                     lat: Optional[float] = None,
                     lon: Optional[float] = None) -> tuple[float, float, Optional[str], Optional[str]]:
    """
    Returns (probability, confidence_score, next_outage_start_iso, next_outage_end_iso).
    Scans FORWARD_STEPS windows ahead to find the first high-risk window.
    solar dict contains live kp/sfi/xray values.
    lat/lon are passed through to the feature builder for solar zenith angle.
    """
    feature_cols  = schema["feature_cols"]
    band_classes  = schema.get("band_classes", [])
    threshold     = schema.get("best_threshold", 0.5)

    # Probability at current time
    row = _build_feature_row(spots, current_time, band_classes, feature_cols, solar, lat, lon)
    proba = float(rf.predict_proba(row)[0][1])

    # Confidence: tree agreement, normalised to [0, 1].
    # For binary classification, per-tree proba ∈ [0,1] so std ∈ [0, 0.5].
    # We normalise against that max so perfect agreement → 1.0, max
    # disagreement → 0.0, and the result is always non-negative.
    tree_preds = np.array([t.predict_proba(row)[0][1] for t in rf.estimators_])
    confidence = float(1.0 - tree_preds.std() / 0.5)
    confidence = max(0.0, min(1.0, confidence))

    # Forward scan to estimate outage window
    outage_start = outage_end = None
    for step in range(1, FORWARD_STEPS + 1):
        future_time = current_time + pd.Timedelta(minutes=step * STEP_MINUTES)
        future_row  = _build_feature_row(spots, future_time, band_classes, feature_cols, solar, lat, lon)
        future_prob = float(rf.predict_proba(future_row)[0][1])
        if future_prob >= threshold:
            if outage_start is None:
                outage_start = future_time
            outage_end = future_time + pd.Timedelta(minutes=STEP_MINUTES)
        elif outage_start is not None:
            break  # outage window ended

    start_iso = outage_start.isoformat() if outage_start else None
    end_iso   = outage_end.isoformat()   if outage_end   else None
    return proba, confidence, start_iso, end_iso


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/")
def root():
    idx = FRONTEND_DIR / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    return {"message": "WSPR Outage Predictor API", "docs": "/docs"}


@app.get("/health")
def health():
    return {
        "status":          "ok",
        "model_loaded":    MODEL_FILE.exists(),
        "features_loaded": FEATURES_FILE.exists(),
        "cache_ready":     _cache_ready,
        "grids_cached":    len(_grid_spot_cache),
    }


@app.get("/stations")
def stations():
    """Return list of known WSPR stations with lat/lon and grid square."""
    idx = _load_grid_index()
    if idx.empty:
        return {"stations": []}
    sub = idx.rename(columns={"tx_lat": "latitude", "tx_lon": "longitude"})
    sub = sub.dropna(subset=["latitude", "longitude"])
    return {"stations": sub.to_dict(orient="records")}


@app.get("/heatmap_data")
def heatmap_data():
    """Return outage probability for all known stations at current time."""
    if not _ensure_cache():
        raise HTTPException(status_code=503,
                            detail="Spot cache is still building. Try again in a moment.")
    try:
        rf     = _load_model()
        schema = _load_schema()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    from pipeline.solar import get_current_solar
    current_time = _effective_time()
    solar        = get_current_solar()
    feature_cols = schema["feature_cols"]
    band_classes = schema.get("band_classes", [])

    # Build all feature rows first, then predict in one batched call
    meta   = []   # (grid, lat, lon) for each valid row
    rows   = []   # feature vectors

    for grid, (lat, lon) in _grid_meta.items():
        spots = _recent_spots(grid, current_time)
        if spots.empty:
            continue
        try:
            row = _build_feature_row(spots, current_time, band_classes, feature_cols, solar, lat, lon)
            rows.append(row[0])   # shape (n_features,)
            meta.append((grid, lat, lon))
        except Exception as e:
            log.warning(f"Heatmap feature build skip {grid}: {e}")

    if not rows:
        return {"heatmap": []}

    # Single batched inference call — orders of magnitude faster than per-row loop
    X     = np.array(rows, dtype=float)
    probs = rf.predict_proba(X)[:, 1]

    results = [
        {
            "grid_square": grid,
            "latitude":    lat,
            "longitude":   lon,
            "probability": round(float(prob), 4),
            "risk_level":  _risk_level(float(prob)),
        }
        for (grid, lat, lon), prob in zip(meta, probs)
    ]

    log.info(f"Heatmap computed: {len(results)} stations")
    return {"heatmap": results}


@app.get("/solar_status")
def solar_status():
    """
    Return current solar and geomagnetic conditions plus any active
    NOAA SWPC alerts, watches, or warnings.
    """
    from pipeline.solar import get_current_solar, get_current_alerts
    solar  = get_current_solar()
    alerts = get_current_alerts()

    # Derive a human-readable storm level from Kp
    kp = solar.get("kp_index", 0)
    if kp >= 9:
        storm_level = "G5 (Extreme)"
    elif kp >= 8:
        storm_level = "G4 (Severe)"
    elif kp >= 7:
        storm_level = "G3 (Strong)"
    elif kp >= 6:
        storm_level = "G2 (Moderate)"
    elif kp >= 5:
        storm_level = "G1 (Minor)"
    else:
        storm_level = "Quiet"

    xray_names = {0: "A", 1: "B", 2: "C", 3: "M", 4: "X"}
    xray_label = xray_names.get(int(solar.get("xray_flux_class", 0)), "A")

    return {
        "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "kp_index":           round(solar.get("kp_index", 0), 2),
        "kp_max_3h":          round(solar.get("kp_max_3h", 0), 2),
        "kp_trend":           round(solar.get("kp_trend", 0), 2),
        "geomagnetic_storm":  bool(solar.get("geomagnetic_storm", 0)),
        "storm_level":        storm_level,
        "sfi":                round(solar.get("sfi", 0), 1),
        "xray_flux_class":    int(solar.get("xray_flux_class", 0)),
        "xray_flux_label":    xray_label,
        "active_alerts":      alerts,
        "alert_count":        len(alerts),
    }


@app.post("/predict_outage", response_model=PredictResponse)
def predict_outage(req: PredictRequest):
    # ── Parse current_time ──────────────────────────────────────────────
    if req.current_time:
        try:
            requested = pd.Timestamp(req.current_time).tz_convert("UTC")
        except Exception:
            raise HTTPException(status_code=422, detail="Invalid current_time format.")
    else:
        requested = None
    current_time = _effective_time(requested)

    # ── Resolve grid square ─────────────────────────────────────────────
    grid_index = _load_grid_index()
    grid = req.grid_square
    lat  = req.latitude
    lon  = req.longitude

    if grid is None and (lat is not None and lon is not None):
        if not grid_index.empty:
            grid = _nearest_grid(lat, lon, grid_index)
    if grid is None:
        raise HTTPException(status_code=422, detail="Provide grid_square or latitude+longitude.")

    # Resolve lat/lon from compact cache if not provided
    if (lat is None or lon is None) and grid in _grid_meta:
        lat, lon = _grid_meta[grid]

    # ── Cache key ───────────────────────────────────────────────────────
    cache_key = f"{grid}|{current_time.floor('5min')}"
    if cache_key in _prediction_cache:
        log.info(f"Cache hit: {cache_key}")
        return _prediction_cache[cache_key]

    # ── Load model ──────────────────────────────────────────────────────
    try:
        rf     = _load_model()
        schema = _load_schema()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # ── Fetch recent spots from compact cache ────────────────────────────
    if not _cache_ready:
        resp = PredictResponse(
            location=LocationInfo(latitude=lat, longitude=lon, grid_square=grid),
            next_outage_start=None, next_outage_end=None,
            probability=0.0, risk_level="low", confidence_score=0.0,
            prediction_unavailable=True,
            unavailable_reason="Spot cache still loading — try again shortly.",
            model_version=schema.get("model_version", "1.0"),
        )
        return resp

    spots = _recent_spots(grid, current_time)

    if spots.empty:
        resp = PredictResponse(
            location=LocationInfo(latitude=lat, longitude=lon, grid_square=grid),
            next_outage_start=None, next_outage_end=None,
            probability=0.0, risk_level="low", confidence_score=0.0,
            prediction_unavailable=True,
            unavailable_reason=f"Insufficient recent data for grid {grid}.",
            model_version=schema.get("model_version", "1.0"),
        )
        log.warning(f"No recent spots for {grid} at {current_time}")
        return resp

    # ── Fetch live solar/geomagnetic conditions ──────────────────────────
    from pipeline.solar import get_current_solar
    solar = get_current_solar()

    # ── Select per-band model if available ──────────────────────────────
    if "band_enc" in spots.columns and not spots["band_enc"].isna().all():
        band_enc_val = int(spots["band_enc"].mode().iloc[0])
        band_rf     = _load_band_model(band_enc_val)
        band_schema = _load_band_schema(band_enc_val)
        if band_rf is not None and band_schema is not None:
            rf     = band_rf
            schema = band_schema
            log.info(f"Using per-band model for band_enc={band_enc_val}")

    # ── Predict ─────────────────────────────────────────────────────────
    try:
        prob, conf, start_iso, end_iso = _predict_outage(rf, schema, spots, current_time, solar, lat, lon)
    except Exception as e:
        log.error(f"Prediction failed for {grid}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    resp = PredictResponse(
        location=LocationInfo(latitude=lat, longitude=lon, grid_square=grid),
        next_outage_start=start_iso,
        next_outage_end=end_iso,
        probability=round(prob, 4),
        risk_level=_risk_level(prob),
        confidence_score=round(conf, 4),
        model_version=schema.get("model_version", "1.0"),
    )

    # Log and cache
    log.info(
        f"PREDICT grid={grid} time={current_time.isoformat()} "
        f"prob={prob:.3f} risk={resp.risk_level} "
        f"next_outage={start_iso}  "
        f"kp={solar.get('kp_index', '?'):.1f}  sfi={solar.get('sfi', '?'):.0f}"
    )
    _prediction_cache[cache_key] = resp
    return resp
