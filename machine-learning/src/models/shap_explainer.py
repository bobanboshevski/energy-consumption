"""
SHAP-based model explainability for the multivariate energy demand LSTM.

Three model variants are explained:
  Keras model:       GradientExplainer — uses TF gradient information (fast)
  ONNX base:         KernelExplainer   — model-agnostic perturbation (slower)
  ONNX quantized:    KernelExplainer   — separate artifact to detect quantization drift

Explanations are computed for every forecast date (same dates as forecast_predictions.json)
and logged to MLflow as separate JSON artifacts per variant.

Artifact structure per variant:
  shap_explanations_keras.json
  shap_explanations_onnx.json
  shap_explanations_onnx_quantized.json

Each artifact contains:
  - feature_importance:  mean |SHAP| per feature (aggregated across all 30 timesteps)
  - timestep_importance: mean |SHAP| per timestep (aggregated across all 4 features)
  - shap_matrix:         raw (30 × 4) SHAP values for heatmap rendering
  - base_value:          expected model output over background (baseline)
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pandas as pd


# ── Background dataset ────────────────────────────────────────────────────────

def _build_background(
        scaled_history: np.ndarray,
        window_size: int,
        n_samples: int = 50,  # todo: what is the purpose of this?
        random_state: int = 42,
) -> np.ndarray:
    """
    Samples n_samples sliding windows from scaled historical data.

    The background represents "typical" model input and serves as the SHAP
    baseline — what the model output would be if we knew nothing about the
    specific input being explained.

    Args:
        scaled_history: All scaled historical rows, shape (n_rows, n_features)
        window_size:    LSTM context window length
        n_samples:      Number of background windows to sample
        random_state:   Random seed for reproducibility

    Returns:
        np.ndarray of shape (n_samples, window_size, n_features)
    """
    max_windows = len(scaled_history) - window_size
    if max_windows <= 0:
        raise ValueError(
            f"Not enough historical data for SHAP background. "
            f"Need > {window_size} rows, got {len(scaled_history)}."
        )

    n_samples = min(n_samples, max_windows)
    rng = np.random.default_rng(random_state)
    indices = rng.choice(max_windows, size=n_samples, replace=False)

    return np.stack([
        scaled_history[i: i + window_size]
        for i in sorted(indices)
    ])  # (n_samples, window_size, n_features)


# ── Forecast window reconstruction ───────────────────────────────────────────

def _reconstruct_forecast_windows(
        predict_fn: Callable[[np.ndarray], np.ndarray],
        scaled_history: np.ndarray,
        forecast_df: pd.DataFrame,
        feature_cols: list[str],
        window_size: int,
        n_features: int,
        feature_scaler,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Replays the autoregressive forecast loop to capture the exact input window
    that was used for each forecast prediction.

    The window at step i contains the previous i predictions appended to the
    historical context — matching exactly what generate_and_log_forecast used.

    Returns:
        np.ndarray of shape (n_forecast, window_size, n_features)
    """
    # scaling the features because we use them when predicting
    forecast_features_scaled = feature_scaler.transform(
        forecast_df[feature_cols].ffill().values
    )

    current_window = scaled_history[-window_size:].copy()
    forecast_windows = []
    scaled_target_preds = []

    for i in range(len(forecast_df)):
        forecast_windows.append(current_window.copy())

        X = current_window.reshape(1, window_size, n_features).astype(np.float32)
        pred_scaled = predict_fn(X)  # (1, 1)
        scaled_target_preds.append(pred_scaled[0])  # (1,)

        new_row = np.concatenate([pred_scaled[0], forecast_features_scaled[i]])
        current_window = np.vstack([current_window[1:], new_row])

    # (n_forecast, window_size, n_features)
    return np.stack(forecast_windows), np.array(scaled_target_preds)


# ── SHAP aggregation helpers ──────────────────────────────────────────────────
def _aggregate_shap_matrix(
        shap_matrix: np.ndarray,
        feature_names: list[str],
) -> tuple[dict, list]:
    """
    Aggregates a (window_size, n_features) SHAP matrix into two summary views:

    feature_importance:  mean |SHAP| per feature, summed across all timesteps.
                         Answers: "which input variable mattered most overall?"
    timestep_importance: mean |SHAP| per timestep, summed across all features.
                         Answers: "which past day mattered most?"

    Returns:
        (feature_importance dict, timestep_importance list)
    """
    # todo: is rows features, and columns timesteps?
    feature_importance = {
        name: round(float(np.mean(np.abs(shap_matrix[:, i]))), 6)
        for i, name in enumerate(feature_names)
    }

    timestep_importance = [
        round(float(np.mean(np.abs(shap_matrix[t, :]))), 6)
        for t in range(shap_matrix.shape[0])
    ]
    return feature_importance, timestep_importance


def _build_explanation_record(
        date: str,
        predicted_demand_gw: float,
        base_value: float,
        shap_matrix: np.ndarray,
        feature_names: list[str],
) -> dict:
    """Builds one explanation record for a single forecast date."""
    feature_importance, timestep_importance = _aggregate_shap_matrix(shap_matrix, feature_names)
    return {
        "date": date,
        "predicted_demand": round(predicted_demand_gw, 4),
        "base_value": round(base_value, 4),  # todo: how is this calculated?
        "feature_importance": feature_importance,
        "timestep_importance": timestep_importance,
        "shap_matrix": shap_matrix.round(6).tolist(),  # (window_size, n_features) matrix of SHAP values (HEATMAP)
    }


# ── Artifact writer ───────────────────────────────────────────────────────────
def _save_shap_artefact(
        explanations: list[dict],
        model_variant: str,
        shap_method: str,
        n_background_samples: int,
        feature_names: list[str],
        window_size: int,
        output_dir: str,
        artifact_name: str,
) -> str:
    """Serializes SHAP explanations to JSON and writes to disk."""
    artifact = {
        "generated_at": datetime.utcnow().isoformat(),
        "model_variant": model_variant,
        "shap_method": shap_method,
        "n_background_samples": n_background_samples,
        "feature_names": feature_names,
        "window_size": window_size,
        "n_explanations": len(explanations),
        "explanations": explanations,
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    artefact_path = Path(output_dir) / artifact_name
    artefact_path.write_text(json.dumps(artifact, indent=2))  # todo: what is indent for?

    size_kb = round(artefact_path.stat().st_size / 1024, 1)
    print(f"[SHAP] Saved {artifact_name} - {len(explanations)} explanations in {size_kb} KB")
    return str(artefact_path)


# ── Variant 1: Keras GradientExplainer ───────────────────────────────────────
def _explain_keras(
        model,
        background: np.ndarray,
        scaled_history: np.ndarray,
        forecast_df: pd.DataFrame,
        forecast_dates: list[str],
        feature_cols: list[str],
        feature_names: list[str],
        window_size: int,
        n_features: int,
        feature_scaler,
        target_scaler,
        output_dir: str,
) -> Optional[str]:
    """
    Computes SHAP values using GradientExplainer for the Keras/TensorFlow model.

    GradientExplainer uses TensorFlow's automatic differentiation to estimate
    the sensitivity of each input to the output. It is 10-100× faster than
    KernelExplainer for neural networks because it does not require perturbation.
    """
    try:
        import shap

        keras_predict_fn = lambda X: model.predict(X, verbose=0)

        # Reconstruct windows and get Keras' own scaled predictions
        forecast_windows, scaled_preds = _reconstruct_forecast_windows(
            predict_fn=keras_predict_fn,
            scaled_history=scaled_history,
            forecast_df=forecast_df,
            feature_cols=feature_cols,
            window_size=window_size,
            n_features=n_features,
            feature_scaler=feature_scaler,
        )

        # Inverse transform: scaled → GW (Keras' own predictions)
        forecast_preds_gw = [
            round(float(target_scaler.inverse_transform([[float(p[0])]])[0][0]), 4)
            for p in scaled_preds
        ]

        print(f"[SHAP:Keras] GradientExplainer — {len(forecast_dates)} dates, "
              f"background: {background.shape}")

        explainer = shap.GradientExplainer(model, background.astype(np.float32))
        base_value = float(np.mean(model.predict(background.astype(np.float32), verbose=0)))

        shap_vals = explainer.shap_values(forecast_windows.astype(np.float32))
        if isinstance(shap_vals, list):
            shap_arr = np.array(shap_vals[0])
        else:
            shap_arr = np.array(shap_vals)

        # Squeeze trailing dim of size 1 if GradientExplainer adds it
        # (TF wraps single-output predictions in an extra dimension)
        if shap_arr.ndim == 4:
            # (n_forecast, window_size, n_features, 1) → (n_forecast, window_size, n_features)
            shap_arr = shap_arr.squeeze(-1)

        explanations = []
        for i, (date, pred_gw) in enumerate(zip(forecast_dates, forecast_preds_gw)):
            shap_matrix = shap_arr[i]
            record = _build_explanation_record(date, pred_gw, base_value, shap_matrix, feature_names)
            top_feat = max(record["feature_importance"], key=record["feature_importance"].get)
            print(f"[SHAP:Keras] {date} → {pred_gw:.4f} GW, top feature: {top_feat}")
            explanations.append(record)

        return _save_shap_artefact(
            explanations=explanations,
            model_variant="keras",
            shap_method="gradient_shap",
            n_background_samples=background.shape[0],
            feature_names=feature_names,
            window_size=window_size,
            output_dir=output_dir,
            artifact_name="shap_explanations_keras.json",
        )

    except Exception as e:
        print(f"[SHAP:Keras] GradientExplainer failed: {e}")
        return None


# ── Variant 2 & 3: ONNX KernelExplainer ──────────────────────────────────────

def _explain_onnx(
        onnx_path: str,
        background: np.ndarray,
        scaled_history: np.ndarray,
        forecast_df: pd.DataFrame,
        forecast_dates: list[str],
        feature_cols: list[str],
        feature_names: list[str],
        window_size: int,
        n_features: int,
        feature_scaler,
        target_scaler,
        output_dir: str,
        artifact_name: str,
        model_variant: str,
        nsamples: int = 100,
) -> Optional[str]:
    """
        Computes SHAP values using KernelExplainer for an ONNX model.

        KernelExplainer is completely model-agnostic — it only needs a predict
        function and a background dataset. Required for ONNX because ONNX Runtime
        does not expose gradients.

        The 3D LSTM input (window_size × n_features) is flattened to 1D for SHAP,
        then reshaped back to 3D before each model call.

        Args:
            nsamples: Coalitions to sample per explained point. Higher = more
                      accurate but slower. 100 is a good balance for CI pipelines.
        """
    try:
        import shap
        import onnxruntime as rt

        print(f"[SHAP:ONNX] Loading: {Path(onnx_path).name}")
        sess_opts = rt.SessionOptions()
        sess_opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = rt.InferenceSession(
            onnx_path,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        input_name = session.get_inputs()[0].name

        # Autoregressive predict_fn: used both for window reconstruction and KernelExplainer
        def onnx_predict_3d(X: np.ndarray) -> np.ndarray:
            """(1, window_size, n_features) → (1, 1)"""
            return session.run(None, {input_name: X.astype(np.float32)})[0]

        def onnx_predict_flat(X_flat: np.ndarray) -> np.ndarray:
            """Flat 2D (n, window × features) → prediction (n,) — for KernelExplainer."""
            X_3d = X_flat.reshape(-1, window_size, n_features).astype(np.float32)
            return session.run(None, {input_name: X_3d})[0].flatten()

        # Reconstruct windows using THIS ONNX variant's own inference
        forecast_windows, scaled_preds = _reconstruct_forecast_windows(
            predict_fn=onnx_predict_3d,
            scaled_history=scaled_history,
            forecast_df=forecast_df,
            feature_cols=feature_cols,
            window_size=window_size,
            n_features=n_features,
            feature_scaler=feature_scaler,
        )

        # Inverse transform: scaled → GW (this ONNX variant's own predictions)
        forecast_preds_gw = [
            round(float(target_scaler.inverse_transform([[float(p[0])]])[0][0]), 4)
            for p in scaled_preds
        ]

        # Flatten for KernelExplainer
        background_flat = background.reshape(background.shape[0], -1)
        forecast_flat = forecast_windows.reshape(forecast_windows.shape[0], -1)

        base_value = float(np.mean(onnx_predict_flat(background_flat)))
        print(f"[SHAP:ONNX] KernelExplainer — background: {background_flat.shape}, "
              f"nsamples={nsamples}")

        explainer = shap.KernelExplainer(onnx_predict_flat, background_flat)
        shap_vals = explainer.shap_values(forecast_flat, nsamples=nsamples, silent=True)

        if isinstance(shap_vals, list):
            shap_arr = np.array(shap_vals[0])
        else:
            shap_arr = np.array(shap_vals)

        explanations = []
        for i, (date, pred_gw) in enumerate(zip(forecast_dates, forecast_preds_gw)):
            shap_matrix = shap_arr[i].reshape(window_size, n_features)
            record = _build_explanation_record(date, pred_gw, base_value, shap_matrix, feature_names)
            top_feat = max(record["feature_importance"], key=record["feature_importance"].get)
            print(f"[SHAP:ONNX] {date} → {pred_gw:.4f} GW, top feature: {top_feat}")
            explanations.append(record)

        return _save_shap_artefact(
            explanations=explanations,
            model_variant=model_variant,
            shap_method="kernel_shap",
            n_background_samples=background.shape[0],
            feature_names=feature_names,
            window_size=window_size,
            output_dir=output_dir,
            artifact_name=artifact_name,
        )

    except Exception as e:
        print(f"[SHAP:ONNX] KernelExplainer failed for {Path(onnx_path).name}: {e}")
        return None


# ── Public entry point ────────────────────────────────────────────────────────
def generate_shap_explanations(
        model,
        pipeline,
        df_full: pd.DataFrame,
        target_col: str,
        feature_cols: list[str],
        window_size: int,
        onnx_paths: dict,
        output_dir: str = "models",
        n_background_samples: int = 50,
        kernel_nsamples: int = 600,
) -> dict:
    """
    Generates SHAP explanations for all available model variants.

    Each variant independently:
    1. Replays the autoregressive forecast loop with its own inference
    2. Records its own predicted_demand values (in GW)
    3. Computes SHAP values explaining its own outputs

    This means predicted_demand will differ slightly between variants,
    correctly reflecting each model's actual output rather than sharing
    the Keras model's predictions.

    Note: forecast_artifact is no longer accepted — forecast dates and
    per-variant predictions are computed directly from df_full.

    Args:
        model:                Trained Keras model
        pipeline:             Fitted sklearn pipeline (for scaling)
        df_full:              Complete dataset including forecast rows
        target_col:           Target column name
        feature_cols:         Feature column names
        window_size:          LSTM context window size
        onnx_paths:           Dict: {'onnx': path, 'onnx_quantized': path}
        output_dir:           Directory to write JSON artifacts
        n_background_samples: SHAP background sample count
        kernel_nsamples:      KernelExplainer perturbation samples
    """
    feature_names = [target_col] + feature_cols
    n_features = len(feature_names)

    # ── Prepare shared data structures ────────────────────────────────────────
    df = df_full.sort_values("Date").reset_index(drop=True)
    history = df[df[target_col].notna()].copy()
    forecast_df = df[df[target_col].isna()].copy().reset_index(drop=True)

    if forecast_df.empty:
        print("[SHAP] No forecast rows found — skipping SHAP generation.")
        return {}

    forecast_dates = forecast_df["Date"].astype(str).tolist()

    preprocess_step = pipeline.named_steps["preprocess"]
    scaled_history = preprocess_step.transform(history[[target_col] + feature_cols])
    feature_scaler = preprocess_step.transformers_[1][1].named_steps["normalize"]
    target_scaler = preprocess_step.transformers_[0][1].named_steps["normalize"]

    # ── Shared SHAP background (same for all variants) ────────────────────────
    background = _build_background(scaled_history, window_size, n_background_samples)

    print(f"[SHAP] {len(forecast_dates)} forecast dates: "
          f"{forecast_dates[0]} → {forecast_dates[-1]}")
    print(f"[SHAP] Background: {background.shape}")

    artifacts = {}

    # ── Keras: GradientExplainer ──────────────────────────────────────────────
    path = _explain_keras(
        model=model,
        background=background,
        scaled_history=scaled_history,
        forecast_df=forecast_df,
        forecast_dates=forecast_dates,
        feature_cols=feature_cols,
        feature_names=feature_names,
        window_size=window_size,
        n_features=n_features,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        output_dir=output_dir,
    )
    if path:
        artifacts["keras"] = path

    # ── ONNX base: KernelExplainer ────────────────────────────────────────────
    if "onnx" in onnx_paths:
        path = _explain_onnx(
            onnx_path=onnx_paths["onnx"],
            background=background,
            scaled_history=scaled_history,
            forecast_df=forecast_df,
            forecast_dates=forecast_dates,
            feature_cols=feature_cols,
            feature_names=feature_names,
            window_size=window_size,
            n_features=n_features,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            output_dir=output_dir,
            artifact_name="shap_explanations_onnx.json",
            model_variant="onnx",
            nsamples=kernel_nsamples,
        )
        if path:
            artifacts["onnx"] = path

    # ── ONNX quantized: KernelExplainer ──────────────────────────────────────
    if "onnx_quantized" in onnx_paths:
        path = _explain_onnx(
            onnx_path=onnx_paths["onnx_quantized"],
            background=background,
            scaled_history=scaled_history,
            forecast_df=forecast_df,
            forecast_dates=forecast_dates,
            feature_cols=feature_cols,
            feature_names=feature_names,
            window_size=window_size,
            n_features=n_features,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            output_dir=output_dir,
            artifact_name="shap_explanations_onnx_quantized.json",
            model_variant="onnx_quantized",
            nsamples=kernel_nsamples,
        )
        if path:
            artifacts["onnx_quantized"] = path

    print(f"[SHAP] Done — generated artifacts: {list(artifacts.keys())}")
    return artifacts
