# FadeWatch

A complete end-to-end machine learning system that predicts HF radio signal fadeouts and outages at any location on Earth, using WSPR (Weak Signal Propagation Reporter) spot data from 2024–2026.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Installation](#2-installation)
3. [Data Preparation](#3-data-preparation)
4. [Running the Pipeline](#4-running-the-pipeline)
5. [Starting the API Server](#5-starting-the-api-server)
6. [Using the Map UI](#6-using-the-map-ui)
7. [API Reference](#7-api-reference)
8. [System Architecture](#8-system-architecture)
9. [Outage Definition](#9-outage-definition)
10. [Model Architecture & Hyperparameters](#10-model-architecture--hyperparameters)
11. [Train / Validation / Test Split](#11-train--validation--test-split)
12. [Feature Engineering](#12-feature-engineering)
13. [Known Limitations](#13-known-limitations)

---

## 1. Project Structure

```
wspr_outage_predictor/
├── data/
│   ├── raw/               ← Place your WSPR CSV files here (2024_01.csv, etc.)
│   └── processed/         ← Generated: unified, labeled, features, splits
├── models/                ← Generated: rf_outage_model.pkl, model_schema.json
├── logs/                  ← Prediction request logs
├── pipeline/
│   ├── ingest.py          ← Stage 1: CSV ingestion & unification
│   ├── label.py           ← Stage 2: Outage event labeling
│   ├── features.py        ← Stage 3: Feature engineering
│   ├── split.py           ← Stage 4: Time-aware data splitting
│   ├── train.py           ← Stage 5: Random Forest training
│   └── preprocess.py      ← Shared inference preprocessing
├── backend/
│   └── app.py             ← FastAPI prediction service
├── frontend/
│   └── index.html         ← Map-based UI (Leaflet + heatmap)
├── run_pipeline.py        ← End-to-end runner
├── requirements.txt
└── README.md
```

---

## 2. Installation

```bash
cd wspr_outage_predictor
pip install -r requirements.txt
```

**Python 3.10+** is required.

---

## 3. Data Preparation

Place your WSPR CSV files inside `data/raw/`.  
Expected filenames: `2024_01.csv`, `2024_02.csv`, … `2026_12.csv`

The ingestion step handles a wide variety of column naming conventions automatically. Supported variants include:

| Canonical name | Accepted aliases |
|---|---|
| `timestamp` | `date`, `time`, `spot_date`, `datetime` |
| `tx_call` | `txcall`, `Tx`, `call` |
| `rx_call` | `rxcall`, `Rx` |
| `snr` | `SNR`, `snr_db` |
| `frequency_hz` | `frequency`, `freq`, `MHz` |
| `tx_grid` | `grid`, `locator`, `tx_locator`, `Maidenhead` |
| `band` | `Band` |

---

## 4. Running the Pipeline

```bash
# Full pipeline (ingest → label → solar → features → split → train)
python run_pipeline.py

# Skip re-ingestion if the unified file already exists
python run_pipeline.py --skip-ingest

# Skip training (e.g., to just regenerate features)
python run_pipeline.py --skip-train
```

### What each stage produces

| Stage | Output file |
|---|---|
| Ingest | `data/processed/wspr_unified.parquet` |
| Label | `data/processed/wspr_labeled.parquet` |
| Solar | `data/processed/solar_kp_cache.parquet`, `solar_sfi_cache.parquet` |
| Features | `data/processed/wspr_features.parquet`, `feature_meta.json` |
| Split | `data/processed/splits/{train,val,test}.parquet`, `split_meta.json` |
| Train | `models/rf_outage_model.pkl`, `models/model_schema.json` |

---

## 5. Starting the API Server

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

The interactive API docs are available at:  
`http://localhost:8000/docs`

The map UI is served at:  
`http://localhost:8000/`

---

## 6. Using the Map UI

Open `http://localhost:8000/` in a browser.

- **Click anywhere** on the map → sends a `/predict_outage` request for that lat/lon.
- **Dropdown** at the top of the side panel → choose a known WSPR grid square.
- **Time slider** → adjust the look-ahead offset (0–120 minutes).
- **Refresh Heatmap** button → overlays risk probability across all known stations.

The side panel shows:
- Location (grid square + coordinates)
- Risk level (green / yellow / red badge)
- Outage probability (percentage + progress bar)
- Next expected outage window (start–end time)
- Confidence score (tree agreement)

---

## 7. API Reference

### `POST /predict_outage`

**Request:**
```json
{
  "latitude":     48.8566,
  "longitude":    2.3522,
  "grid_square":  "JN03",
  "current_time": "2026-03-15T14:00:00Z"
}
```
`grid_square` is optional if `latitude`+`longitude` are provided.  
`current_time` defaults to now (UTC) if omitted.

**Response:**
```json
{
  "location": {
    "latitude":    48.8566,
    "longitude":   2.3522,
    "grid_square": "JN03"
  },
  "next_outage_start":   "2026-03-15T14:40:00+00:00",
  "next_outage_end":     "2026-03-15T15:00:00+00:00",
  "probability":         0.72,
  "risk_level":          "high",
  "confidence_score":    0.88,
  "prediction_unavailable": false,
  "model_version":       "1.0"
}
```

| `risk_level` | Probability range |
|---|---|
| `low` | 0 – 32 % |
| `medium` | 33 – 65 % |
| `high` | 66 – 100 % |

### `GET /health`
Returns model and data loading status.

### `GET /stations`
Returns a list of all known WSPR grid squares with their lat/lon centroids.

### `GET /heatmap_data`
Returns outage probability for all known stations at the current time.

---

## 8. System Architecture

```
CSV Files (2024-2026)
      │
      ▼
 [ingest.py]  ─── normalise columns, parse UTC timestamps,
                   convert Maidenhead → lat/lon, dedup, save parquet
      │
      ▼
 [label.py]   ─── define outage events per grid square
                   (gap threshold, low-SNR, silence window)
                   → outage_label, future_outage_label
      │
      ▼
 [solar.py]   ─── fetch Kp index + F10.7 SFI from NOAA SWPC
                   merge onto WSPR data by timestamp
                   cache to solar_kp_cache.parquet / solar_sfi_cache.parquet
      │
      ▼
 [features.py] ── rolling SNR, gap, spot-density, band,
                   temporal + solar/geomagnetic features per grid × time window
      │
      ▼
 [split.py]   ─── chronological 60/20/20 split
                   (train ≤ train_end, val, test)
      │
      ▼
 [train.py]   ─── RandomForestClassifier(n=250, depth=12, balanced)
                   threshold tuning on val set → model_schema.json
      │
      ▼
 [app.py]     ─── FastAPI: POST /predict_outage
                   fetches recent spots + live Kp/SFI from NOAA,
                   builds feature vector, runs model,
                   returns probability + outage window
      │
      ▼
 [index.html] ─── Leaflet map UI, heatmap overlay, side panel
```

---

## 9. Outage Definition

An outage is flagged at grid square `G` at time `t` if **any** of the following hold:

| Condition | Default threshold | Tuneable in |
|---|---|---|
| Time gap between consecutive spots | > **30 minutes** | `label.py: GAP_THRESHOLD_MINUTES` |
| SNR below threshold for N consecutive spots | < **−28 dB** for **3** spots | `label.py: SNR_THRESHOLD`, `SUSTAINED_SPOTS` |
| Zero spots received in rolling window | **30-minute** window | `label.py: SILENCE_WINDOW_MINUTES` |

The **target variable** (`future_outage_label`) is `1` if any outage starts within the next **60 minutes** of time `t`.

---

## 10. Model Architecture & Hyperparameters

| Parameter | Value |
|---|---|
| Algorithm | `RandomForestClassifier` (scikit-learn) |
| `n_estimators` | 250 |
| `max_depth` | 12 |
| `min_samples_leaf` | 10 |
| `class_weight` | `"balanced"` (handles class imbalance) |
| `random_state` | 42 |
| `n_jobs` | −1 (all CPU cores) |

Probability threshold is **tuned on the validation set** to maximise F1 on the outage class, then applied at inference time.

---

## 11. Train / Validation / Test Split

| Split | Fraction | Approx. date range |
|---|---|---|
| Train | 60 % | Jan 2024 – mid-2025 |
| Validation | 20 % | mid-2025 – late 2025 |
| Test | 20 % | 2026 |

Exact boundaries are computed chronologically from the data timestamps and written to `data/processed/splits/split_meta.json` after the split stage.  
**No random shuffling is applied** — splits are strictly ordered in time to prevent data leakage.

---

## 12. Feature Engineering

| Feature | Description |
|---|---|
| `snr_rolling_mean/min/max/std` | Rolling stats over the last 5 spots |
| `snr_diff` | SNR change from previous spot |
| `snr_slope` | Linear SNR trend over last 5 spots (negative = degrading) |
| `time_gap_min` | Minutes since last spot |
| `gap_rolling_max/min/mean` | Rolling gap stats |
| `gap_acceleration` | Rate of change of rolling mean gap (positive = widening) |
| `spots_per_10min` | Spot count in 10-min window |
| `spots_per_30min` | Spot count in 30-min window |
| `spot_rate_trend` | Recent 30-min density minus prior 30-min density |
| `mins_since_last_outage` | Minutes since last recorded outage at this grid |
| `frequency_hz` | Transmission frequency |
| `band_enc` | Integer-encoded amateur band |
| `hour_of_day` | UTC hour (0–23) |
| `day_of_week` | Day (0=Mon, 6=Sun) |
| `is_daytime` | 1 if 06:00–20:00 UTC |
| `kp_index` | Instantaneous planetary Kp index (0–9) from NOAA SWPC |
| `kp_max_3h` | Maximum Kp in the preceding 3 hours |
| `sfi` | Daily F10.7 solar flux index from NOAA SWPC |
| `kp_x_sfi` | Interaction term: `kp_index × sfi / 100` |

---

## 13. Known Limitations

- **Sparse coverage**: Many grid squares have very few WSPR spots. Predictions for rare/remote grids may have low confidence or return `prediction_unavailable`.
- **Static model**: The model is trained on historical data up to the training cutoff. It does not auto-retrain on new spots. Retrain periodically by re-running `run_pipeline.py`.
- **Solar data coverage**: NOAA SWPC's free endpoints only provide ~3 days of Kp history. For training, Kp is merged from the cache captured at pipeline run time; spots before the cache window receive the quiet-day default (Kp=2, SFI=120). Consider sourcing a historical Kp archive (e.g. GFZ Potsdam) for a more accurate training set.
- **Temporal resolution**: Features are computed per spot (not per fixed time grid), so dense-spot grids have better time resolution than sparse ones.
- **Outage definition is heuristic**: The gap/SNR/silence thresholds are sensible defaults but should be validated by a domain expert for each deployment.
- **Class imbalance**: Outage events are naturally rare. The `class_weight="balanced"` setting compensates, but may not be sufficient in very imbalanced datasets; consider SMOTE or upsampling if needed.
