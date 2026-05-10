"""
Generates forecast predictions for all null-energy_demand rows
and saves them as a JSON artifact in the active MLflow run.

Called from train.py after training completes. This caches predictions
so the backend can serve them without re-running the model on every request.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path


def generate_and_log_forecast(
        model,
        pipeline,
        df_full: pd.DataFrame,
        target_col: str,
        feature_cols: list,
        window_size: int,
        output_dir: str = "models",
) -> dict:
    """
    Generates predictions for all rows where energy_demand is null
    (both delayed rows and genuine future forecast rows).

    Args:
        model:       Trained Keras model
        pipeline:    Fitted sklearn Pipeline
        df_full:     Complete dataset (historical + forecast rows)
        target_col:  Name of the target column
        feature_cols: List of feature column names
        window_size: LSTM context window size
        output_dir:  Directory to write the JSON artifact

    Returns:
        Dict with predictions and metadata (also saved to disk for MLflow logging)
    """

    df = df_full.sort_values("Date").reset_index(drop=True)

    # Historical = rows with known energy_demand
    history = df[df[target_col].notna()].copy()

    # Forecast = rows where energy_demand is null (delayed data + future)
    forecast = df[df[target_col].isna()].copy().reset_index(drop=True)

    if forecast.empty:
        print("No forecast rows found — skipping forecast artifact generation.")
        return {}

    # ── Scale historical data using the fitted pipeline preprocessor ──────────
    all_cols = [target_col] + feature_cols
    preprocess_step = pipeline.named_steps["preprocess"]
    scaled_history = preprocess_step.transform(history[all_cols])

    target_scaler = preprocess_step.transformers_[0][1].named_steps["normalize"]
    feature_scaler = preprocess_step.transformers_[1][1].named_steps["normalize"]

    # ── Autoregressive prediction loop ────────────────────────────────────────
    window = scaled_history[-window_size:].copy()
    forecast_features = forecast[feature_cols].ffill().values
    forecast_features_scaled = feature_scaler.transform(forecast_features)

    predictions = []

    for i in range(len(forecast)):
        X = window.reshape(1, window_size, len(all_cols))
        pred_scaled = model.predict(X, verbose=0)
        pred_value = float(target_scaler.inverse_transform(pred_scaled)[0][0])

        # Slide window: drop oldest day, append this prediction
        new_row = np.concatenate([pred_scaled[0], forecast_features_scaled[i]])
        window = np.vstack([window[1:], new_row])

        predictions.append({
            "date": str(forecast["Date"].iloc[i]),
            "predicted_demand": round(pred_value, 4),
            "is_confirmed": bool(not forecast["is_forecast"].iloc[i]),
            "temp_max": float(forecast["temp_max"].iloc[i]) if pd.notna(forecast["temp_max"].iloc[i]) else None,
            "temp_min": float(forecast["temp_min"].iloc[i]) if pd.notna(forecast["temp_min"].iloc[i]) else None,
            "daylight_duration": float(forecast["daylight_duration"].iloc[i]) if pd.notna(
                forecast["daylight_duration"].iloc[i]) else None,
        })

    # ── Build artifact payload ─────────────────────────────────────────────────
    artifact = {
        "generated_at": datetime.utcnow().isoformat(),
        "last_historical_date": str(history["Date"].iloc[-1]),
        "forecast_dates": [p["date"] for p in predictions],
        "predictions": predictions,
    }

    # ── Write to disk so MLflow can log it ────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    artifact_path = Path(output_dir) / "forecast_predictions.json"
    artifact_path.write_text(json.dumps(artifact, indent=2))
    print(f"Forecast artifact saved: {artifact_path} ({len(predictions)} predictions)")

    return artifact
