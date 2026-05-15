import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta

from app.core.config import settings
from app.core.data_service import get_data
from app.core.model_loader import load_univariate_model, load_univariate_pipeline, predict_univariate


def predict_for_date(target_date: date) -> dict:
    """
    Predicts energy demand for a specific future date using the univariate LSTM model.

    The model only uses historical energy_demand values (no weather features),
    so it can predict arbitrarily far into the future by feeding each prediction
    back as input for the next step (autoregressive forecasting).

    Args:
        target_date: The date to predict energy demand for.

    Returns:
        A dict with the predicted demand and metadata.

    Raises:
        ValueError: If the target date is in the past or beyond the 365-day horizon.
    """
    today = date.today()

    # ── Validation ────────────────────────────────────────────────────────────
    if target_date <= today:
        raise ValueError(
            f"Target date must be in the future. Got: {target_date}, today is: {today}"
        )

    days_ahead = (target_date - today).days

    if days_ahead > settings.UNIVARIATE_MAX_HORIZON_DAYS:
        raise ValueError(
            f"Cannot predict more than {settings.UNIVARIATE_MAX_HORIZON_DAYS} days ahead. "
            f"Requested {days_ahead} days ahead ({target_date}). "
            f"Maximum date: {today + timedelta(days=settings.UNIVARIATE_MAX_HORIZON_DAYS)}"
        )

    # ── Load models ───────────────────────────────────────────────────────────
    model = load_univariate_model()
    pipeline = load_univariate_pipeline()

    # ── Load and prepare historical data ──────────────────────────────────────
    df = get_data()
    history = df[df[settings.TARGET_COL].notna()].copy()
    history = history.sort_values("Date").reset_index(drop=True)

    if len(history) < settings.UNIVARIATE_WINDOW_SIZE:
        raise ValueError(
            f"Not enough historical data to build the context window. "
            f"Need {settings.UNIVARIATE_WINDOW_SIZE} rows, got {len(history)}"
        )

    # ── Scale historical data ─────────────────────────────────────────────────
    # Use only the target column — univariate model takes shape (window_size, 1)
    history_values = history[[settings.TARGET_COL]].values
    scaler = pipeline.named_steps["normalize"]
    scaled_history = scaler.transform(history_values)

    # ── Autoregressive prediction ─────────────────────────────────────────────
    # Start with the last WINDOW_SIZE days as context
    window = scaled_history[-settings.UNIVARIATE_WINDOW_SIZE:].copy()

    # Predict day by day until we reach the target date
    last_historical_date = pd.to_datetime(history["Date"].iloc[-1]).date()  # todo: check this
    current_date = last_historical_date

    intermediate_predictions = []

    while current_date < target_date:
        X = window.reshape(1, settings.UNIVARIATE_WINDOW_SIZE, 1)
        pred_scaled = model.predict(X, verbose=0)

        # Slide window forward
        window = np.vstack([window[1:], pred_scaled[0].reshape(1, 1)])

        current_date += timedelta(days=1)
        pred_value = float(scaler.inverse_transform(pred_scaled)[0][0])
        intermediate_predictions.append({
            "date": str(current_date),
            "predicted_demand": round(pred_value, 4),
        })

    # The last prediction is for the target date
    final = intermediate_predictions[-1]

    return {
        "target_date": str(target_date),
        "predicted_demand": final["predicted_demand"],
        "demand_category": _classify_demand(final["predicted_demand"]),
        "days_ahead": days_ahead,
        "model": "univariate_lstm",
        "last_known_date": str(last_historical_date),
        "note": (
            "Univariate model — uses only historical energy demand patterns. "
            "Accuracy decreases for dates further in the future."
        ),
    }


def predict_range(start_date: date, end_date: date) -> list:
    """
    Predicts energy demand for every day in a date range.
    Useful for displaying a long-range forecast chart on the frontend.

    Args:
        start_date: First day to predict (must be in the future).
        end_date: Last day to predict (max 365 days from today).

    Returns:
        List of daily predictions.
    """
    today = date.today()

    if start_date <= today:
        raise ValueError(f"Start date must be in the future. Got: {start_date}")

    if end_date < start_date:
        raise ValueError(f"End date must be after start date.")

    days_ahead = (end_date - today).days
    if days_ahead > settings.UNIVARIATE_MAX_HORIZON_DAYS:
        raise ValueError(
            f"Cannot predict more than {settings.UNIVARIATE_MAX_HORIZON_DAYS} days ahead. "
            f"Requested end date {end_date} is {days_ahead} days away."
        )

    load_univariate_model()  # ensures ONNX or Keras is loaded
    pipeline = load_univariate_pipeline()

    df = get_data()
    history = df[df[settings.TARGET_COL].notna()].copy()
    history = history.sort_values("Date").reset_index(drop=True)

    history_values = history[[settings.TARGET_COL]].values
    scaler = pipeline.named_steps["normalize"]
    scaled_history = scaler.transform(history_values)

    window = scaled_history[-settings.UNIVARIATE_WINDOW_SIZE:].copy()

    last_historical_date = pd.to_datetime(history["Date"].iloc[-1]).date()
    current_date = last_historical_date

    results = []

    while current_date < end_date:
        X = window.reshape(1, settings.UNIVARIATE_WINDOW_SIZE, 1)
        pred_scaled = predict_univariate(X.astype(np.float32))
        window = np.vstack([window[1:], pred_scaled[0].reshape(1, 1)])

        current_date += timedelta(days=1)
        pred_value = float(scaler.inverse_transform(pred_scaled)[0][0])

        if current_date >= start_date:
            results.append({
                "date": str(current_date),
                "predicted_demand": round(pred_value, 4),
                "demand_category": _classify_demand(pred_value),
                "days_ahead": (current_date - today).days,
            })

    return results


def _classify_demand(value: float) -> str:
    if value < settings.DEMAND_LOW_THRESHOLD:
        return "low"
    elif value < settings.DEMAND_HIGH_THRESHOLD:
        return "medium"
    else:
        return "high"
