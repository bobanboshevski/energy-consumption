import os
import sys
import tempfile
from pathlib import Path
import mlflow
import mlflow.tensorflow
import joblib
from tensorflow import keras
from app.core import onnx_runner
import numpy as np

from app.core.activate_model import (
    get_active_version, set_loaded_version,
    MODEL_MULTIVARIATE, MODEL_UNIVARIATE,
)
from app.core.config import settings
from app.core.mlflow_client import get_latest_version

sys.path.insert(0, str(Path(__file__).resolve().parent))

_model = None
_pipeline = None
_univariate_model = None
_univariate_pipeline = None


def _setup_mlflow():
    os.environ["MLFLOW_TRACKING_USERNAME"] = settings.MLFLOW_TRACKING_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = settings.MLFLOW_TRACKING_PASSWORD
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)


def resolve_version(model_key: str, model_name: str) -> str:
    active = get_active_version(model_key)
    if active != "latest":
        return active
    latest = get_latest_version(model_name)
    return latest if latest is not None else "1"


def _load_pipeline_from_mlflow(model_name: str, version: str, artifact_filename: str):
    _setup_mlflow()
    client = mlflow.tracking.MlflowClient()
    run_id = client.get_model_version(model_name, version).run_id
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact_filename,
            dst_path=tmp_dir,
        )
        return joblib.load(local_path)


def load_model():
    global _model

    if _model is not None or onnx_runner.is_loaded():
        return _model  # None is fine if ONNX session is loaded

    version = resolve_version(MODEL_MULTIVARIATE, settings.MLFLOW_MODEL_NAME)
    print(f"Loading multivariate model with version {version}")

    # Try ONNX first
    try:
        _setup_mlflow()
        client = mlflow.tracking.MlflowClient()
        run_id = client.get_model_version(settings.MLFLOW_MODEL_NAME, version).run_id
        onnx_loaded = onnx_runner.load_onnx_session(run_id, version, model_key=MODEL_MULTIVARIATE,
                                                    prefer_quantized=True)
    except Exception as e:
        print(f"[ONNX] ⚠  Could not attempt ONNX load: {e}")
        onnx_loaded = False

    if onnx_loaded:
        set_loaded_version(version, MODEL_MULTIVARIATE)
        print(f"[MODEL] ✓  Using ONNX Runtime for inference (v{version})")
        return None  # callers must use predict_multivariate()

    # Fall back to Keras
    try:
        _setup_mlflow()
        _model = mlflow.tensorflow.load_model(
            f"models:/{settings.MLFLOW_MODEL_NAME}/{version}"
        )
        print(f"Multivariate model loaded from MLflow (v{version})")
    except Exception as e:
        print(f"WARNING: MLflow load failed ({e}). Falling back to local file.")
        _model = keras.models.load_model(settings.MODEL_PATH)
        version = "local"

    set_loaded_version(version, MODEL_MULTIVARIATE)
    return _model


def predict_multivariate(X: np.ndarray) -> np.ndarray:
    """
    Unified inference for the multivariate model.
    Uses ONNX Runtime if loaded, otherwise Keras.
    Callers never need to know which backend is active.

    X shape: (1, window_size, n_features)
    Returns: (1, 1)
    """
    if onnx_runner.is_loaded():
        return onnx_runner.predict(X)
    model = load_model()
    return model.predict(X, verbose=1)


def load_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    version = resolve_version(MODEL_MULTIVARIATE, settings.MLFLOW_MODEL_NAME)

    try:
        _pipeline = _load_pipeline_from_mlflow(
            settings.MLFLOW_MODEL_NAME, version, "pipeline_energy_demand.pkl"
        )
        print(f"Multivariate pipeline loaded from MLflow (v{version})")
    except Exception as e:
        print(f"WARNING: MLflow pipeline load failed ({e}). Falling back to local.")
        _pipeline = joblib.load(settings.PIPELINE_PATH)

    return _pipeline


# ── Univariate ────────────────────────────────────────────────────────────────

def load_univariate_model():
    global _univariate_model
    if _univariate_model is not None:
        return _univariate_model

    version = resolve_version(MODEL_UNIVARIATE, settings.MLFLOW_UNIVARIATE_MODEL_NAME)

    try:
        _setup_mlflow()
        client = mlflow.tracking.MlflowClient()
        run_id = client.get_model_version(settings.MLFLOW_UNIVARIATE_MODEL_NAME, version).run_id
        onnx_loaded = onnx_runner.load_onnx_session(run_id, version, model_key=MODEL_UNIVARIATE, prefer_quantized=True)

        # _univariate_model = mlflow.tensorflow.load_model(
        #     f"models:/{settings.MLFLOW_UNIVARIATE_MODEL_NAME}/{version}"
        # )
        print(f"Univariate model loaded from MLflow (v{version})")
    except Exception as e:
        print(f"[ONNX:univariate] ⚠  Could not attempt ONNX load: {e}")
        onnx_loaded = False

    if onnx_loaded:
        set_loaded_version(version, MODEL_UNIVARIATE)
        print(f"[MODEL] ✓  Using ONNX Runtime for univariate inference (v{version})")
        return None

    # Fall back to Keras
    try:
        _setup_mlflow()
        _univariate_model = mlflow.tensorflow.load_model(
            f"models:/{settings.MLFLOW_UNIVARIATE_MODEL_NAME}/{version}"
        )
        print(f"Univariate model loaded from MLflow (v{version})")

    except Exception as e:
        print(f"WARNING: MLflow univariate load failed ({e}). Falling back to local.")
        _univariate_model = keras.models.load_model(
            settings.UNIVARIATE_MODEL_PATH)  # todo: this in production cannot be satisfied, since i build the backend independently
        # todo: from the /machine-learning directory ! ! !
        version = "local"

    set_loaded_version(version, MODEL_UNIVARIATE)
    return _univariate_model


def load_univariate_pipeline():
    global _univariate_pipeline
    if _univariate_pipeline is not None:
        return _univariate_pipeline

    version = resolve_version(MODEL_UNIVARIATE, settings.MLFLOW_UNIVARIATE_MODEL_NAME)

    try:
        _univariate_pipeline = _load_pipeline_from_mlflow(
            settings.MLFLOW_UNIVARIATE_MODEL_NAME, version, "pipeline_univariate.pkl"
        )
        print(f"Univariate pipeline loaded from MLflow (v{version})")
    except Exception as e:
        print(f"WARNING: MLflow univariate pipeline load failed ({e}). Falling back to local.")
        _univariate_pipeline = joblib.load(settings.UNIVARIATE_PIPELINE_PATH)

    return _univariate_pipeline


def predict_univariate(X: np.ndarray) -> np.ndarray:
    """
    Unified inference for the univariate model.
    Uses ONNX Runtime if loaded, otherwise Keras.

    X shape: (1, window_size, 1)
    Returns: (1, 1)
    """
    if onnx_runner.is_loaded("univariate"):
        return onnx_runner.predict(X, model_key="univariate")
    model = load_univariate_model()
    return model.predict(X, verbose=0)


# ── Utilities ─────────────────────────────────────────────────────────────────

def reload_models():
    """Clears in-memory models so next request reloads from MLflow."""
    global _model, _pipeline, _univariate_model, _univariate_pipeline
    _model = None
    _pipeline = None
    _univariate_model = None
    _univariate_pipeline = None
    onnx_runner.clear()  # clear ONNX session too
    print("Models cleared — will reload on next request.")
