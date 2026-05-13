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

_onnx_session = None
_onnx_loaded_version: Optional[str] = None


def _setup_mlflow():
    import mlflow
    from app.core.config import settings
    os.environ["MLFLOW_TRACKING_USERNAME"] = settings.MLFLOW_TRACKING_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = settings.MLFLOW_TRACKING_PASSWORD
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)


def _download_onnx_artifact(run_id: str, prefer_quantized: bool = True) -> Optional[str]:
    """
    Downloads the ONNX model artifact from MLflow.
    Tries quantized version first, falls back to base ONNX.
    """
    import mlflow

    candidates = (
        ["model_energy_demand_quantized.onnx", "model_energy_demand.onnx"]
        if prefer_quantized
        else ["model_energy_demand.onnx"]
    )

    for filename in candidates:
        try:
            tmp_dir = tempfile.mkdtemp()
            local_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path=filename,
                dst_path=tmp_dir,
            )
            print(f"[ONNX] ✓  Downloaded: {filename}")
            return local_path
        except Exception:
            continue

    return None


def load_onnx_session(run_id: str, version: str) -> bool:
    """
    Loads the ONNX Runtime session for the given MLflow run.
    Returns True if successful, False if ONNX is unavailable.
    """
    global _onnx_session, _onnx_loaded_version

    try:
        import onnxruntime as rt
    except ImportError:
        print("[ONNX] ⚠  onnxruntime not installed — falling back to Keras inference")
        return False

    _setup_mlflow()
    onnx_path = _download_onnx_artifact(run_id)

    if not onnx_path:
        print("[ONNX] —  No ONNX artifact found for this model version — using Keras")
        return False

    try:
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

        _onnx_session = rt.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        _onnx_loaded_version = version

        input_name = _onnx_session.get_inputs()[0].name
        input_shape = _onnx_session.get_inputs()[0].shape
        print(f"[ONNX] ✓  Session loaded (v{version}) | input: {input_name} {input_shape}")
        return True

    except Exception as e:
        print(f"[ONNX] ✗  Failed to create ONNX session: {e}")
        return False


def predict(X: np.ndarray) -> np.ndarray:
    """
    Runs inference using the ONNX Runtime session.
    X shape: (1, window_size, n_features)
    Returns shape: (1, 1)
    """
    if _onnx_session is None:
        raise RuntimeError("ONNX session not loaded. Call load_onnx_session() first.")

    input_name = _onnx_session.get_inputs()[0].name
    result = _onnx_session.run(None, {input_name: X.astype(np.float32)})
    return result[0]


def is_loaded() -> bool:
    return _onnx_session is not None


def get_loaded_version() -> Optional[str]:
    return _onnx_loaded_version


def clear():
    global _onnx_session, _onnx_loaded_version
    _onnx_session = None
    _onnx_loaded_version = None
    print("[ONNX] Session cleared")
