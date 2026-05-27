from sklearn.metrics import mean_squared_error, mean_absolute_error

from app.core import onnx_runner
from app.core.config import settings
import pandas as pd
import numpy as np

from app.core.data_service import get_data, get_drift_report_html, get_gx_report_html
from app.core.model_loader import load_model, load_pipeline, load_univariate_model, load_univariate_pipeline, \
    predict_multivariate, resolve_version, _setup_mlflow, predict_univariate


# ── Multivariate model monitoring ─────────────────────────────────────────────

def get_model_performance_over_time(window_days: int = 30) -> list:
    """
    Evaluates the model on historical data, sampling ~20 evenly spaced points
    within the last `window_days` days.

    Returns a list of {date, actual, predicted, error} dicts.
    """
    # THIS WAS THE OLD WAY - READING FROM CSV FILE
    # df = pd.read_csv(settings.DATA_PATH)
    df = get_data()

    # Only use historical rows where we have the real target value
    df = df[df["is_forecast"] == False].copy()
    df = df[df[settings.TARGET_COL].notna()].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Limit to the last `window_days` + WINDOW_SIZE rows
    # (we need extra rows before the evaluation window to build the initial context window)
    eval_start = max(0, len(df) - window_days - settings.WINDOW_SIZE)
    df = df.iloc[eval_start:].reset_index(drop=True)

    # model = load_model()
    load_model()  # ensure loaded (ONNX or Keras)

    pipeline = load_pipeline()

    all_cols = [settings.TARGET_COL] + settings.FEATURE_COLS

    # Use only the preprocessor step (not the sliding window)
    # so we get raw scaled values to build windows manually
    preprocess_step = pipeline.named_steps["preprocess"]
    scaled = preprocess_step.transform(df[all_cols])

    # Get target scaler for inverse transform
    target_scaler = preprocess_step.transformers_[0][1].named_steps["normalize"]

    results = []

    # Sample ~20 evenly spaced evaluation points from the evaluation window
    # (starting after the initial context window of WINDOW_SIZE days)
    eval_count = min(20, len(df) - settings.WINDOW_SIZE)
    step = max(1, (len(df) - settings.WINDOW_SIZE) // eval_count)

    # i starts at WINDOW_SIZE so we always have a full window of history before it
    for i in range(settings.WINDOW_SIZE, len(df), step):
        # Window = the WINDOW_SIZE rows immediately before position i
        window = scaled[i - settings.WINDOW_SIZE:i]
        X = window.reshape(1, settings.WINDOW_SIZE, len(all_cols))

        # pred_scaled = model.predict(X, verbose=1)
        pred_scaled = predict_multivariate(X.astype(np.float32))

        # Inverse transform: go from scaled [0,1] back to GW values
        pred_value = float(target_scaler.inverse_transform(pred_scaled)[0][0])

        # iloc[i] = get the actual energy demand at position i in the dataframe
        actual = float(df[settings.TARGET_COL].iloc[i])

        results.append({
            "date": df["Date"].iloc[i].strftime("%Y-%m-%d"),
            "actual": round(actual, 4),
            "predicted": round(pred_value, 4),
            "error": round(abs(actual - pred_value), 4),
        })

    return results


def get_current_metrics(window_days: int = 30) -> dict:
    """Computes MAE, MSE, RMSE for the multivariate model."""
    perf = get_model_performance_over_time(window_days)
    if not perf:
        return {}
    return _compute_metrics(perf)


# ── Univariate model monitoring ───────────────────────────────────────────────
def get_univariate_performance_over_time(window_days: int = 30) -> list:
    """
    Evaluates the univariate model on historical data.
    Uses only energy_demand column — no weather features.
    Samples ~20 evenly spaced points within the last `window_days` days.
    """
    df = get_data()
    df = df[df["is_forecast"] == False].copy()
    df = df[df[settings.TARGET_COL].notna()].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    window_size = settings.UNIVARIATE_WINDOW_SIZE
    eval_start = max(0, len(df) - window_days - window_size)
    df = df.iloc[eval_start:].reset_index(drop=True)

    load_univariate_model()
    pipeline = load_univariate_pipeline()

    # Univariate: single column, simple pipeline
    values = df[[settings.TARGET_COL]].values
    scaler = pipeline.named_steps["normalize"]
    scaled = scaler.transform(values)

    results = []
    eval_count = min(20, len(df) - window_size)
    step = max(1, (len(df) - window_size) // eval_count)

    for i in range(window_size, len(df), step):
        window = scaled[i - window_size:i]
        X = window.reshape(1, window_size, 1)
        pred_scaled = predict_univariate(X.astype(np.float32))
        pred_value = float(scaler.inverse_transform(pred_scaled)[0][0])
        actual = float(df[settings.TARGET_COL].iloc[i])

        results.append({
            "date": df["Date"].iloc[i].strftime("%Y-%m-%d"),
            "actual": round(actual, 4),
            "predicted": round(pred_value, 4),
            "error": round(abs(actual - pred_value), 4),
        })

    return results


def get_univariate_metrics(window_days: int = 30) -> dict:
    """Computes MAE, MSE, RMSE for the univariate model."""
    perf = get_univariate_performance_over_time(window_days)
    if not perf:
        return {}
    return _compute_metrics(perf)


# ── Drift ─────────────────────────────────────────────────────────────────────

def get_drift_report_summary() -> dict:
    """Returns metadata about the latest Evidently drift report."""
    try:
        html = get_drift_report_html()
        if html is None:
            return {"available": False, "reason": "Report not found on DagShub or locally"}

        size_kb = round(len(html.encode("utf-8")) / 1024, 1)
        return {
            "available": True,
            "size_kb": size_kb,
            "source": "dagshub",
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


# ── Great expectations ───────────────────────────────────────────────────
def get_gx_report_summary() -> dict:
    """Returns metadata about the latest Great Expectations validation report."""
    try:
        html = get_gx_report_html()
        if html is None:
            return {"available": False, "reason": "Report not found on DagShub or locally"}
        return {
            "available": True,
            "size_kb": round(len(html.encode("utf-8")) / 1024, 1),
            "source": "dagshub",
            "passed": "PASSED" in html,
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


# ── Shared helpers ────────────────────────────────────────────────────────────

def _compute_metrics(perf: list) -> dict:
    errors = [float(p["error"]) for p in perf]
    actuals = [float(p["actual"]) for p in perf]
    preds = [float(p["predicted"]) for p in perf]

    mae = float(mean_absolute_error(actuals, preds))
    mse = float(mean_squared_error(actuals, preds))

    return {
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(float(np.sqrt(mse)), 4),
        "mean_error": round(float(np.mean(errors)), 4),
        "max_error": round(float(np.max(errors)), 4),
        "data_points": len(perf),
    }


# ── Backend comparison (ONNX vs Keras) ───────────────────────────────────────

def _evaluate_with_fn(
        scaled: np.ndarray,
        df: pd.DataFrame,
        window_size: int,
        n_features: int,
        target_scaler,
        predict_fn,
        eval_count: int = 20,
) -> list:
    """
    Generic evaluation loop — runs `predict_fn` on evenly spaced historical points.
    Works with any backend (ONNX, Keras, quantized).
    """
    results = []
    step = max(1, (len(df) - window_size) // eval_count)

    for i in range(window_size, len(df), step):
        window = scaled[i - window_size:i]
        X = window.reshape(1, window_size, n_features).astype(np.float32)

        try:
            pred_scaled = predict_fn(X)
            pred_value = float(target_scaler.inverse_transform(pred_scaled)[0][0])
            actual = float(df[settings.TARGET_COL].iloc[i])
            results.append({
                "date": df["Date"].iloc[i].strftime("%Y-%m-%d"),
                "actual": round(actual, 4),
                "predicted": round(pred_value, 4),
                "error": round(abs(actual - pred_value), 4),
            })
        except Exception as e:
            print(f"[COMPARISON] Skipping point {i}: {e}")

    return results


def _load_keras_for_comparison(model_key: str):
    """
    Loads a Keras model directly from MLflow for comparison.
    Does NOT touch global state — model is discarded after comparison.
    """
    import mlflow.tensorflow

    _setup_mlflow()

    if model_key == "multivariate":
        version = resolve_version("multivariate", settings.MLFLOW_MODEL_NAME)
        uri = f"models:/{settings.MLFLOW_MODEL_NAME}/{version}"
    else:
        version = resolve_version("univariate", settings.MLFLOW_UNIVARIATE_MODEL_NAME)
        uri = f"models:/{settings.MLFLOW_UNIVARIATE_MODEL_NAME}/{version}"

    try:
        model = mlflow.tensorflow.load_model(uri)
        print(f"[COMPARISON] Keras model loaded for comparison (v{version})")
        return model
    except Exception as e:
        print(f"[COMPARISON] Could not load Keras model for comparison: {e}")
        return None


def get_backend_comparison(window_days: int = 30, model_key: str = "multivariate") -> dict:
    """
    Evaluates the same historical data with both ONNX and Keras backends.
    Returns side-by-side performance and metrics for direct comparison.

    This is used on the admin UI to visualise accuracy and speed differences
    between the quantized ONNX model and the original Keras model.
    """
    # ── Prepare shared data ───────────────────────────────────────────────────
    df = get_data()
    df = df[df["is_forecast"] == False].copy()
    df = df[df[settings.TARGET_COL].notna()].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    if model_key == "multivariate":
        window_size = settings.WINDOW_SIZE
        all_cols = [settings.TARGET_COL] + settings.FEATURE_COLS
        n_features = len(all_cols)

        load_model()
        pipeline = load_pipeline()

        eval_start = max(0, len(df) - window_days - window_size)
        df = df.iloc[eval_start:].reset_index(drop=True)

        preprocess_step = pipeline.named_steps["preprocess"]
        scaled = preprocess_step.transform(df[all_cols])
        target_scaler = preprocess_step.transformers_[0][1].named_steps["normalize"]

    else:
        window_size = settings.UNIVARIATE_WINDOW_SIZE
        n_features = 1

        load_univariate_model()
        pipeline = load_univariate_pipeline()

        eval_start = max(0, len(df) - window_days - window_size)
        df = df.iloc[eval_start:].reset_index(drop=True)

        scaler = pipeline.named_steps["normalize"]
        scaled = scaler.transform(df[[settings.TARGET_COL]].values)
        target_scaler = scaler

    # ── ONNX evaluation ───────────────────────────────────────────────────────
    onnx_result = {"available": False, "variant": None, "performance": [], "metrics": {}}

    if onnx_runner.is_loaded(model_key):
        # Detect which variant is loaded (quantized vs base)
        variant = "quantized"  # prefer_quantized=True is the default
        predict_fn = lambda X: onnx_runner.predict(X, model_key=model_key)
        perf = _evaluate_with_fn(scaled, df, window_size, n_features, target_scaler, predict_fn)
        onnx_result = {
            "available": True,
            "variant": variant,
            "performance": perf,
            "metrics": _compute_metrics(perf) if perf else {},
        }
        print(f"[COMPARISON] ONNX ({variant}) evaluation complete — {len(perf)} points")
    else:
        print(f"[COMPARISON] ONNX not available for {model_key}")

    # ── Keras evaluation ──────────────────────────────────────────────────────
    keras_result = {"available": False, "performance": [], "metrics": {}}
    keras_model = _load_keras_for_comparison(model_key)

    if keras_model is not None:
        predict_fn = lambda X: keras_model.predict(X, verbose=0)
        perf = _evaluate_with_fn(scaled, df, window_size, n_features, target_scaler, predict_fn)
        keras_result = {
            "available": True,
            "performance": perf,
            "metrics": _compute_metrics(perf) if perf else {},
        }
        print(f"[COMPARISON] Keras evaluation complete — {len(perf)} points")
        del keras_model  # free memory immediately

    return {
        "model_key": model_key,
        "window_days": window_days,
        "onnx": onnx_result,
        "keras": keras_result,
    }
