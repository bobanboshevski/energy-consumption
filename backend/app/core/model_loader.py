import os
import sys
import tempfile
from pathlib import Path
import mlflow
import mlflow.tensorflow
import joblib
from tensorflow import keras

from app.core.activate_model import (
    get_active_version, set_loaded_version,
    MODEL_MULTIVARIATE, MODEL_UNIVARIATE,
)
from app.core.config import settings

_SHARED_DIR = Path(__file__).resolve().parent.parent.parent.parent / "shared"
sys.path.insert(0, str(_SHARED_DIR))

_model = None
_pipeline = None
_univariate_model = None
_univariate_pipeline = None


def _setup_mlflow():
    os.environ["MLFLOW_TRACKING_USERNAME"] = settings.MLFLOW_TRACKING_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = settings.MLFLOW_TRACKING_PASSWORD
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)


def _resolve_version(model_key: str, model_name: str) -> str:
    active = get_active_version(model_key)
    if active != "latest":
        return active
    try:
        _setup_mlflow()
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            return max(versions, key=lambda v: int(v.version)).version
    except Exception as e:
        print(f"WARNING: Could not resolve latest version for {model_name}: {e}")
    return "1"


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


# ── Multivariate ──────────────────────────────────────────────────────────────

def load_model():
    global _model
    if _model is not None:
        return _model

    version = _resolve_version(MODEL_MULTIVARIATE, settings.MLFLOW_MODEL_NAME)
    print(f"Loading multivariate model with version {version}")

    try:
        _setup_mlflow()
        _model = mlflow.tensorflow.load_model(f"models:/{settings.MLFLOW_MODEL_NAME}/{version}")
        print(f"Multivariate model loaded from MLflow (v{version})")
    except Exception as e:
        print(f"WARNING: MLflow load failed ({e}). Falling back to local file.")
        _model = keras.models.load_model(settings.MODEL_PATH)
        version = "local"

    set_loaded_version(version, MODEL_MULTIVARIATE)
    return _model


def load_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    version = _resolve_version(MODEL_MULTIVARIATE, settings.MLFLOW_MODEL_NAME)

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

    version = _resolve_version(MODEL_UNIVARIATE, settings.MLFLOW_UNIVARIATE_MODEL_NAME)

    try:
        _setup_mlflow()
        _univariate_model = mlflow.tensorflow.load_model(
            f"models:/{settings.MLFLOW_UNIVARIATE_MODEL_NAME}/{version}"
        )
        print(f"Univariate model loaded from MLflow (v{version})")
    except Exception as e:
        print(f"WARNING: MLflow univariate load failed ({e}). Falling back to local.")
        _univariate_model = keras.models.load_model(settings.UNIVARIATE_MODEL_PATH)
        version = "local"

    set_loaded_version(version, MODEL_UNIVARIATE)
    return _univariate_model


def load_univariate_pipeline():
    global _univariate_pipeline
    if _univariate_pipeline is not None:
        return _univariate_pipeline

    version = _resolve_version(MODEL_UNIVARIATE, settings.MLFLOW_UNIVARIATE_MODEL_NAME)

    try:
        _univariate_pipeline = _load_pipeline_from_mlflow(
            settings.MLFLOW_UNIVARIATE_MODEL_NAME, version, "pipeline_univariate.pkl"
        )
        print(f"Univariate pipeline loaded from MLflow (v{version})")
    except Exception as e:
        print(f"WARNING: MLflow univariate pipeline load failed ({e}). Falling back to local.")
        _univariate_pipeline = joblib.load(settings.UNIVARIATE_PIPELINE_PATH)

    return _univariate_pipeline


# ── Utilities ─────────────────────────────────────────────────────────────────

def reload_models():
    """Clears in-memory models so next request reloads from MLflow."""
    global _model, _pipeline, _univariate_model, _univariate_pipeline
    _model = None
    _pipeline = None
    _univariate_model = None
    _univariate_pipeline = None
    print("Models cleared — will reload on next request.")
