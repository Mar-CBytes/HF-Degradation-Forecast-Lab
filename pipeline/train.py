"""
Step 5 – Random Forest Model Training
---------------------------------------
Trains a Random Forest classifier on the feature-engineered dataset
to predict future_outage_label.

Workflow:
  1. Load features dataset and split meta.
  2. Select train / val / test rows by timestamp boundaries.
  3. Fit RandomForestClassifier on train set (global model + per-band models).
  4. Tune probability threshold on validation set (maximise F1-outage).
  5. Evaluate on test set; report precision/recall/F1/ROC-AUC.
  6. Persist models, thresholds, and feature schemas to models/.

Per-band models
---------------
80m / 40m / 30m / 20m have very different ionospheric failure modes:
  - 80m  (1.8 MHz): D-layer absorption during daytime; geomagnetic storms
  - 40m  (7 MHz):   Both D-layer and F-layer; most used by WSPR operators
  - 30m  (10 MHz):  Relatively stable; good baseline
  - 20m  (14 MHz):  F-layer dependent; disrupted by X-ray flares (SIDs)

A per-band model is trained when enough minority-class samples exist
(MIN_BAND_OUTAGES). Fall back to the global model at inference when
a band-specific model is unavailable.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
MODELS_DIR    = Path(__file__).resolve().parents[1] / "models"
FEATURES_FILE = PROCESSED_DIR / "wspr_features.parquet"
SPLIT_META    = PROCESSED_DIR / "splits" / "split_meta.json"
FEATURE_META  = PROCESSED_DIR / "feature_meta.json"

MODEL_FILE    = MODELS_DIR / "rf_outage_model.pkl"
SCHEMA_FILE   = MODELS_DIR / "model_schema.json"

# Target column (must match feature_meta.json)
TARGET_COL = "future_outage_label"

# Per-band model file name templates  ({band} = band_enc integer)
BAND_MODEL_STEM  = "rf_outage_model_band_{band}.pkl"
BAND_SCHEMA_STEM = "model_schema_band_{band}.json"

# Minimum outage-class samples in the training split to justify a per-band model
MIN_BAND_OUTAGES = 50

# Undersample majority class to this ratio (majority : minority) before fitting.
# At ~0.7% outage rate, 10:1 still leaves ~120 K training rows — plenty for RF.
UNDERSAMPLE_RATIO = 10  # majority rows per minority row

RF_PARAMS = {
    "n_estimators":  250,
    "max_depth":     12,
    "min_samples_leaf": 10,
    "class_weight":  "balanced",
    "random_state":  42,
    "n_jobs":        -1,
}


def _load_data():
    df = pd.read_parquet(FEATURES_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    with open(SPLIT_META) as f:
        meta = json.load(f)
    with open(FEATURE_META) as f:
        feat_meta = json.load(f)

    train_end = pd.Timestamp(meta["train_end"])
    val_end   = pd.Timestamp(meta["val_end"])

    feature_cols  = feat_meta["feature_cols"]
    target_col    = feat_meta["target_col"]

    # Only keep features that actually exist in the dataframe
    feature_cols = [c for c in feature_cols if c in df.columns]

    train = df[df["timestamp"] <= train_end]
    val   = df[(df["timestamp"] > train_end) & (df["timestamp"] <= val_end)]
    test  = df[df["timestamp"] > val_end]

    log.info(f"Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")
    return train, val, test, feature_cols, target_col, meta, feat_meta


def _best_threshold(y_true, y_prob):
    """Find probability threshold that maximises F1 on the outage class."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = np.where(
        (precisions + recalls) == 0,
        0,
        2 * precisions * recalls / (precisions + recalls),
    )
    best_idx = np.argmax(f1s[:-1])          # thresholds has one fewer element
    return float(thresholds[best_idx]), float(f1s[best_idx])


def _train_one(X_train, y_train, X_val, y_val, X_test, y_test,
               feature_cols, label: str) -> tuple:
    """
    Train, tune, and evaluate a single RF model.
    Returns (rf, schema_dict).
    """
    n_minority = int(y_train.sum())
    n_majority = int(len(y_train) - n_minority)
    log.info(f"  [{label}] Before undersample — majority:{n_majority:,} minority:{n_minority:,}")

    rus = RandomUnderSampler(sampling_strategy=1 / UNDERSAMPLE_RATIO, random_state=42)
    X_tr, y_tr = rus.fit_resample(X_train, y_train)
    log.info(f"  [{label}] After  undersample — {len(X_tr):,} rows")

    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_tr, y_tr)

    val_probs  = rf.predict_proba(X_val)[:, 1]
    best_thresh, val_f1 = _best_threshold(y_val, val_probs)

    test_probs = rf.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= best_thresh).astype(int)
    log.info("\n" + classification_report(y_test, test_preds,
                                          target_names=["no_outage", "outage"],
                                          zero_division=0))
    try:
        auc = roc_auc_score(y_test, test_probs)
    except ValueError:
        auc = None
    test_f1 = f1_score(y_test, test_preds, zero_division=0)
    log.info(f"  [{label}] val_F1={val_f1:.3f}  test_F1={test_f1:.3f}  ROC-AUC={auc or 'n/a'}")

    importances = dict(zip(feature_cols, rf.feature_importances_.tolist()))
    schema = {
        "feature_cols":    feature_cols,
        "target_col":      TARGET_COL,
        "best_threshold":  best_thresh,
        "rf_params":       RF_PARAMS,
        "undersample_ratio": UNDERSAMPLE_RATIO,
        "evaluation": {"val_f1": val_f1, "test_f1": test_f1, "roc_auc": auc},
        "feature_importances": importances,
        "label": label,
    }
    return rf, schema


def train():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df, feature_cols, target_col, split_meta, feat_meta = _load_data()
    band_classes = feat_meta.get("band_classes", [])

    # ── Global model ──────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("GLOBAL MODEL")
    log.info("=" * 60)
    rf, schema = _train_one(
        train_df[feature_cols].values, train_df[target_col].values,
        val_df[feature_cols].values,   val_df[target_col].values,
        test_df[feature_cols].values,  test_df[target_col].values,
        feature_cols, label="global",
    )
    schema["band_classes"]  = band_classes
    schema["split_meta"]    = split_meta
    schema["model_version"] = "1.2"

    joblib.dump(rf, MODEL_FILE)
    log.info(f"Global model saved → {MODEL_FILE}")
    with open(SCHEMA_FILE, "w") as f:
        json.dump(schema, f, indent=2)
    log.info(f"Global schema saved → {SCHEMA_FILE}")

    # ── Per-band models ───────────────────────────────────────────────
    # band_enc column holds the LabelEncoder integer for the band.
    # We iterate over each unique value and train a dedicated model.
    band_results = {}
    if "band_enc" in train_df.columns:
        unique_bands = sorted(train_df["band_enc"].dropna().unique().astype(int).tolist())
        log.info(f"\nPer-band training for {len(unique_bands)} bands: {unique_bands}")

        for band_val in unique_bands:
            tr_b  = train_df[train_df["band_enc"] == band_val]
            va_b  = val_df[val_df["band_enc"]     == band_val]
            te_b  = test_df[test_df["band_enc"]   == band_val]

            n_out = int(tr_b[target_col].sum())
            if n_out < MIN_BAND_OUTAGES:
                log.info(f"  Band enc={band_val}: only {n_out} outage rows — skipping (< {MIN_BAND_OUTAGES})")
                continue
            if va_b[target_col].sum() == 0 or te_b[target_col].sum() == 0:
                log.info(f"  Band enc={band_val}: no outage samples in val or test — skipping")
                continue

            # Band name from LabelEncoder classes list
            band_name = band_classes[band_val] if band_val < len(band_classes) else str(band_val)
            log.info(f"\n{'='*60}\nBAND MODEL: enc={band_val}  name={band_name}\n{'='*60}")

            rf_b, schema_b = _train_one(
                tr_b[feature_cols].values, tr_b[target_col].values,
                va_b[feature_cols].values, va_b[target_col].values,
                te_b[feature_cols].values, te_b[target_col].values,
                feature_cols, label=f"band_{band_name}",
            )
            schema_b["band_enc_value"] = band_val
            schema_b["band_name"]      = band_name
            schema_b["band_classes"]   = band_classes
            schema_b["model_version"]  = "1.2"

            model_path  = MODELS_DIR / BAND_MODEL_STEM.format(band=band_val)
            schema_path = MODELS_DIR / BAND_SCHEMA_STEM.format(band=band_val)
            joblib.dump(rf_b, model_path)
            with open(schema_path, "w") as f:
                json.dump(schema_b, f, indent=2)
            log.info(f"  Band model saved → {model_path.name}")

            band_results[band_val] = {
                "band_name": band_name,
                "test_f1":   schema_b["evaluation"]["test_f1"],
                "roc_auc":   schema_b["evaluation"]["roc_auc"],
            }

    # ── Summary ───────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("TRAINING SUMMARY")
    log.info("=" * 60)
    log.info(f"  Global   test_F1={schema['evaluation']['test_f1']:.4f}  "
             f"ROC-AUC={schema['evaluation']['roc_auc']:.4f}")
    for bv, br in band_results.items():
        log.info(f"  Band {br['band_name']:4s}  test_F1={br['test_f1']:.4f}  "
                 f"ROC-AUC={br['roc_auc'] or 'n/a'}")
    log.info("=" * 60)

    return rf, schema


if __name__ == "__main__":
    train()
