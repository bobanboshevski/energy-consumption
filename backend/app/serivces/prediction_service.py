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
    """Returns predictions for future rows (is_forecast=True) where energy_demand is unknown."""
    model = load_model()
    pipeline = load_pipeline()

    # df = pd.read_csv(settings.DATA_PATH)
    df = get_data()
    df = df.sort_values("Date").reset_index(drop=True)

    # Get historical data to build the initial window
    # history = df[df["is_forecast"] == False].copy()
    # history = history[history[settings.TARGET_COL].notna()].copy()
    # history = history.sort_values("Date")

    # Historical = rows where we have the real energy_demand value
    # todo: here a potential issue can be if some older value is null or something :/ ???
    history = df[df[settings.TARGET_COL].notna()].copy()

    # Get future rows to predict
    # forecast = df[df["is_forecast"] == True].copy()
    # forecast = forecast.sort_values("Date")

    # Forecast = ALL rows where energy_demand is null
    # This includes is_forecast=False rows where data is just delayed
    forecast = df[df[settings.TARGET_COL].isna()].copy()
    forecast = forecast.reset_index(drop=True)

    if forecast.empty:
        return []

    for col in settings.FEATURE_COLS:
        if history[col].isna().all():
            raise ValueError(f"Feature column '{col}' is entirely null in historical data")

    # Scale the historical data using the saved fitted pipeline's preprocessor only
    # (no sliding window — we need raw scaled values to build windows manually)
    all_cols = [settings.TARGET_COL] + settings.FEATURE_COLS
    history_features = history[all_cols].copy()

    preprocess_step = pipeline.named_steps["preprocess"]
    # we do fit_transform only when training the model, so we don't need to do it here
    scaled_history = preprocess_step.transform(history_features)

    target_scaler = preprocess_step.transformers_[0][1].named_steps["normalize"]
    feature_scaler = preprocess_step.transformers_[1][1].named_steps["normalize"]

    # Use last WINDOW_SIZE rows as initial context window
    window = scaled_history[-settings.WINDOW_SIZE:].copy()

    # Scale forecast weather features
    forecast_features = forecast[settings.FEATURE_COLS].ffill().values
    forecast_features_scaled = feature_scaler.transform(forecast_features)

    predictions = []

    for i in range(len(forecast)):
        # Reshape window to (1, window_size, num_features)
        X = window.reshape(1, settings.WINDOW_SIZE, len(all_cols))

        # Predict next day (scaled)
        pred_scaled = model.predict(X, verbose=1)

        # Inverse transform to get actual GW value
        pred_value = float(target_scaler.inverse_transform(pred_scaled)[0][0])

        # Build the new row to append to window:
        # target column (predicted, scaled) + weather features (scaled)
        new_row = np.concatenate([pred_scaled[0], forecast_features_scaled[i]])
        print(f"new_row: {new_row} for predicting {forecast['Date'].iloc[i]}")
        window = np.vstack([window[1:], new_row])  # slide window by 1

        # Classify demand category using second model
        # Uncomment when classifier is ready:
        # category_features = forecast[settings.FEATURE_COLS].iloc[i:i+1]
        # cat_idx = _classifier.predict(category_features)[0]
        # category = _label_encoder.inverse_transform([cat_idx])[0]
        category = _classify_demand_simple(pred_value)

        # TODO: I want to change the location of this
        # im storing predicted data, so later when i receive the actual data i can compare it with the predicted data
        _save_predictions(predictions)

        predictions.append({
            "date": forecast["Date"].iloc[i],
            "predicted_demand": round(pred_value, 4),
            "demand_category": category,

            "is_confirmed": bool(not forecast["is_forecast"].iloc[i]),  # True if data just delayed
            "temp_max": float(forecast["temp_max"].iloc[i]) if pd.notna(forecast["temp_max"].iloc[i]) else None,
            "temp_min": float(forecast["temp_min"].iloc[i]) if pd.notna(forecast["temp_min"].iloc[i]) else None,
            "daylight_duration": float(forecast["daylight_duration"].iloc[i]) if pd.notna(
                forecast["daylight_duration"].iloc[i]) else None,
        })

    _save_predictions(predictions)
    return predictions


# todo: i will add a better location for this
_PREDICTIONS_LOG = Path(settings.DATA_PATH).parent.parent / "predictions_log.json"


def _save_predictions(predictions: list):
    """Saves predictions with timestamp so we can compare with actuals later."""
    log = []
    if _PREDICTIONS_LOG.exists():
        with open(_PREDICTIONS_LOG) as f:
            log = json.load(f)

    entry = {
        "generated_at": datetime.utcnow().isoformat(),
        "predictions": predictions
    }
    log.append(entry)

    # todo: Keep only last 90 entries
    # log = log[-90:]

    with open(_PREDICTIONS_LOG, "w") as f:
        json.dump(log, f, indent=2)


# def _classify_demand_simple(value: float) -> str:
#     # todo: this is temporary, i will later develop the second model
#     """
#     Simple rule-based fallback classifier until the ML classifier is ready.
#     Remove this when classifier is uncommented above.
#     """
#     if value < 1.2:
#         return "low"
#     elif value < 1.6:
#         return "medium"
#     else:
#         return "high"

def _classify_demand_simple(value: float) -> str:
    if value < settings.DEMAND_LOW_THRESHOLD:
        return "low"
    elif value < settings.DEMAND_HIGH_THRESHOLD:
        return "medium"
    else:
        return "high"
