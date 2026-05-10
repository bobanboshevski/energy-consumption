import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
from app.core.config import settings
from app.core.data_service import get_data
from app.core.model_loader import load_model, load_pipeline
from datetime import datetime

from app.core.predictions_cache import get_cached_predictions, save_live_cache


def get_historical_data(days: int = 90) -> list:
    """Returns the last `days` rows of historical (non-forecast) data with known target."""
    # df = pd.read_csv(settings.DATA_PATH)
    df = get_data()
    df = df[df["is_forecast"] == False].copy()
    df = df[df[settings.TARGET_COL].notna()].copy()
    df = df.sort_values("Date")
    df = df.tail(days)

    # .to_dict(orient="records") converts the DataFrame to a list of dicts
    # orient="records" means: [{col1: val1, col2: val2}, {col1: val3...}]
    return df[["Date", settings.TARGET_COL] + settings.FEATURE_COLS].to_dict(orient="records")


def get_forecast_data() -> list:
    """
    Returns predictions for all rows where energy_demand is null.

    Priority:
    1. MLflow artifact cache (fast — no model inference needed)
    2. Live inference (slow — runs the LSTM model)

    The cache is invalidated automatically when:
    - A new model version is activated
    - New forecast dates appear in the dataset (daily data update)
    """
    df = get_data()
    df = df.sort_values("Date").reset_index(drop=True)

    forecast = df[df[settings.TARGET_COL].isna()].copy().reset_index(drop=True)

    if forecast.empty:
        return []

    current_forecast_dates = forecast["Date"].tolist()

    # ── Try MLflow artifact cache ──────────────────────────────────────────────
    cached = get_cached_predictions(current_forecast_dates)
    if cached is not None:
        result = _enrich_with_category(cached)
        _save_predictions_log(result)
        return result

    # ── Fall back to live inference ────────────────────────────────────────────
    print("Running live inference (cache miss or unavailable)...")
    return _run_live_inference(df, forecast)


# _ means private function
def _run_live_inference(df: pd.DataFrame, forecast: pd.DataFrame) -> list:
    """Runs the LSTM model to generate predictions. Called only on cache miss."""

    print(f"[INFER] →  Starting live inference for {len(forecast)} date(s): "
          f"{forecast['Date'].iloc[0]} → {forecast['Date'].iloc[-1]}")

    model = load_model()
    pipeline = load_pipeline()

    history = df[df[settings.TARGET_COL].notna()].copy()

    for col in settings.FEATURE_COLS:
        if history[col].isna().all():
            raise ValueError(f"Feature column '{col}' is entirely null in historical data")

    all_cols = [settings.TARGET_COL] + settings.FEATURE_COLS
    preprocess_step = pipeline.named_steps["preprocess"]
    scaled_history = preprocess_step.transform(history[all_cols])

    target_scaler = preprocess_step.transformers_[0][1].named_steps["normalize"]
    feature_scaler = preprocess_step.transformers_[1][1].named_steps["normalize"]

    window = scaled_history[-settings.WINDOW_SIZE:].copy()
    forecast_features = forecast[settings.FEATURE_COLS].ffill().values
    forecast_features_scaled = feature_scaler.transform(forecast_features)

    predictions = []

    for i in range(len(forecast)):
        # Reshape window to (1, window_size, num_features)
        X = window.reshape(1, settings.WINDOW_SIZE, len(all_cols))

        # Predict next day (scaled)
        pred_scaled = model.predict(X, verbose=0)

        # Inverse transform to get actual GW value
        pred_value = float(target_scaler.inverse_transform(pred_scaled)[0][0])

        # Build the new row to append to window:
        # target column (predicted, scaled) + weather features (scaled)
        new_row = np.concatenate([pred_scaled[0], forecast_features_scaled[i]])
        # print(f"new_row: {new_row} for predicting {forecast['Date'].iloc[i]}")
        window = np.vstack([window[1:], new_row])

        predictions.append({
            "date": forecast["Date"].iloc[i],
            "predicted_demand": round(pred_value, 4),
            "is_confirmed": bool(not forecast["is_forecast"].iloc[i]),
            "temp_max": float(forecast["temp_max"].iloc[i]) if pd.notna(forecast["temp_max"].iloc[i]) else None,
            "temp_min": float(forecast["temp_min"].iloc[i]) if pd.notna(forecast["temp_min"].iloc[i]) else None,
            "daylight_duration": float(forecast["daylight_duration"].iloc[i]) if pd.notna(
                forecast["daylight_duration"].iloc[i]) else None,
        })

    # _save_predictions_log(predictions)

    # Save to local cache so subsequent requests skip inference
    save_live_cache(predictions, forecast["Date"].tolist())

    print(f"[INFER] ✓  Live inference complete — {len(predictions)} predictions generated")

    return _enrich_with_category(predictions)


def _enrich_with_category(predictions: list) -> list:
    """Adds demand_category to each prediction dict."""
    return [
        {**p, "demand_category": _classify_demand_simple(p["predicted_demand"])}
        for p in predictions
    ]


# todo: i will add a better location for this
_PREDICTIONS_LOG = Path(__file__).resolve().parent.parent.parent / "predictions_log.json"


def _save_predictions_log(predictions: list):
    """Saves live inference predictions with timestamp for later accuracy evaluation."""
    log = []
    if _PREDICTIONS_LOG.exists():
        try:
            with open(_PREDICTIONS_LOG) as f:
                log = json.load(f)
        except Exception:
            log = []

    log.append({
        "generated_at": datetime.utcnow().isoformat(),
        "predictions": predictions,
    })
    log = log[-90:]  # keep last 90 entries

    try:
        with open(_PREDICTIONS_LOG, "w") as f:
            json.dump(log, f, indent=2)
    except Exception as e:
        print(f"WARNING: Could not save predictions log: {e}")


# def _save_predictions(predictions: list):
#     """Saves predictions with timestamp so we can compare with actuals later."""
#     log = []
#     if _PREDICTIONS_LOG.exists():
#         with open(_PREDICTIONS_LOG) as f:
#             log = json.load(f)
#
#     entry = {
#         "generated_at": datetime.utcnow().isoformat(),
#         "predictions": predictions
#     }
#     log.append(entry)
#
#     # todo: Keep only last 90 entries
#     # log = log[-90:]
#
#     with open(_PREDICTIONS_LOG, "w") as f:
#         json.dump(log, f, indent=2)


def _classify_demand_simple(value: float) -> str:
    if value < settings.DEMAND_LOW_THRESHOLD:
        return "low"
    elif value < settings.DEMAND_HIGH_THRESHOLD:
        return "medium"
    else:
        return "high"
