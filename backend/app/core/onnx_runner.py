import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

"""
ONNX Runtime inference wrapper for the multivariate LSTM model.

Priority for inference:
  1. Quantized ONNX  (fastest, smallest)
  2. Base ONNX       (fast, full precision)
  3. Keras fallback  (handled by model_loader.py)

ONNX Runtime is 2-4× faster than TensorFlow for CPU inference on small models
because it skips TF's graph construction overhead and uses optimized kernels.
"""

# _onnx_session = None
# _onnx_loaded_version: Optional[str] = None

_sessions: dict = {
    "multivariate": None,
    "univariate": None,
}

# Quantized variant is always tried first when prefer_quantized=True
_ONNX_FILENAMES = {
    "multivariate": {
        "quantized": "model_energy_demand_quantized.onnx",
        "base": "model_energy_demand.onnx",
    },
    "univariate": {
        "quantized": "model_energy_demand_univariate_quantized.onnx",
        "base": "model_energy_demand_univariate.onnx",
    },
}


def _setup_mlflow():
    import mlflow
    from app.core.config import settings
    os.environ["MLFLOW_TRACKING_USERNAME"] = settings.MLFLOW_TRACKING_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = settings.MLFLOW_TRACKING_PASSWORD
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)


def _download_onnx_artifact(
        run_id: str,
        model_key: str = "multivariate",
        prefer_quantized: bool = True,
) -> Optional[str]:
    """
    Downloads the ONNX model artifact from MLflow.
    Tries quantized version first, falls back to base ONNX.
    """
    import mlflow

    filenames = _ONNX_FILENAMES[model_key]
    candidates = (
        [filenames["quantized"], filenames["base"]]
        if prefer_quantized
        else [filenames["base"]]
    )

    for filename in candidates:
        try:
            tmp_dir = tempfile.mkdtemp()
            local_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path=filename,
                dst_path=tmp_dir,
            )
            print(f"[ONNX:{model_key}] ✓  Downloaded: {filename}")
            return local_path
        except Exception:
            continue

    return None


def load_onnx_session(
        run_id: str,
        version: str,
        model_key: str = "multivariate",
        prefer_quantized: bool = True,
) -> bool:
    """
    Loads the ONNX Runtime session for the given MLflow run.
    Returns True if successful, False if ONNX is unavailable.
    """
    # global _onnx_session, _onnx_loaded_version

    try:
        import onnxruntime as rt
    except ImportError:
        print(f"[ONNX:{model_key}] ⚠  onnxruntime not installed — falling back to Keras")
        return False

    _setup_mlflow()
    onnx_path = _download_onnx_artifact(run_id, model_key, prefer_quantized)

    if not onnx_path:
        print(f"[ONNX:{model_key}] —  No ONNX artifact for this version — using Keras")
        return False

    try:
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

        _sessions[model_key] = rt.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        # _onnx_loaded_version = version

        input_name = _sessions[model_key].get_inputs()[0].name
        input_shape = _sessions[model_key].get_inputs()[0].shape
        quantized_label = "quantized" if prefer_quantized else "base"

        print(f"[ONNX:{model_key}] ✓  Session loaded (v{version}, {quantized_label}) "
              f"| input: {input_name} {input_shape}"
              )
        return True

    except Exception as e:
        print(f"[ONNX:{model_key}] ✗  Failed to create session: {e}")
        return False


def predict(X: np.ndarray, model_key: str = "multivariate") -> np.ndarray:
    """
    Runs inference using the loaded ONNX session.
    Args:
        X: Input array, shape (1, window_size, n_features)
        model_key: 'multivariate' or 'univariate'
    Returns:
        Output array, shape (1, 1)
    """

    session = _sessions.get(model_key)
    if session is None:
        raise RuntimeError(
            f"ONNX session for '{model_key}' is not loaded. "
            "Call load_onnx_session() first."
        )

    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: X.astype(np.float32)})
    return result[0]


def is_loaded(model_key: str = "multivariate") -> bool:
    return _sessions.get(model_key) is not None


# def get_loaded_version() -> Optional[str]:
#     return _onnx_loaded_version


# def clear():
#     global _onnx_session, _onnx_loaded_version
#     _onnx_session = None
#     _onnx_loaded_version = None
#     print("[ONNX] Session cleared")

def clear(model_key: Optional[str] = None):
    """
    Clears one or all ONNX sessions.

    Args:
        model_key: If provided, clears only that model's session.
                   If None, clears all sessions.
    """
    if model_key:
        _sessions[model_key] = None
        print(f"[ONNX:{model_key}] Session cleared")
    else:
        for key in _sessions:
            _sessions[key] = None
        print("[ONNX] All sessions cleared")
