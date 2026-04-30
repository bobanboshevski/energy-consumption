import os
import sys
import tempfile
from pathlib import Path
import mlflow
import mlflow.tensorflow
import joblib
from tensorflow import keras

from app.core.activate_model import get_active_version
from app.core.config import settings

# Make shared preprocess module available for pipeline unpickling
_SHARED_DIR = Path(__file__).resolve().parent.parent.parent.parent / "shared"
sys.path.insert(0, str(_SHARED_DIR))

_model = None
_pipeline = None
_loaded_version = None


def _setup_mlflow():
    os.environ["MLFLOW_TRACKING_USERNAME"] = settings.MLFLOW_TRACKING_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = settings.MLFLOW_TRACKING_PASSWORD
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)


def _resolve_version() -> str:
    """Gets the version to load — either a specific number or finds the latest."""
    active = get_active_version()
    if active != "latest":
        return active

    try:
        _setup_mlflow()
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{settings.MLFLOW_MODEL_NAME}'")
        if versions:
            latest = max(versions, key=lambda v: int(v.version))
            return latest.version
    except Exception as e:
        print(f"WARNING: Could not resolve latest version: {e}")

    return "1"


def load_model():
    global _model, _loaded_version
    if _model is not None:
        return _model

    version = _resolve_version()

    try:
        _setup_mlflow()
        model_uri = f"models:/{settings.MLFLOW_MODEL_NAME}/{version}"
        _model = mlflow.tensorflow.load_model(model_uri)
        _loaded_version = version
        print(f"Model loaded from MLflow: {model_uri} (version {version})")
    except Exception as e:
        print(f"WARNING: Could not load model from MLflow ({e}). Falling back to local file.")
        _model = keras.models.load_model(settings.MODEL_PATH)
        _loaded_version = "local"

    return _model


def load_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    version = _resolve_version()

    try:
        _setup_mlflow()
        client = mlflow.tracking.MlflowClient()

        model_version = client.get_model_version(settings.MLFLOW_MODEL_NAME, version)
        run_id = model_version.run_id

        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path="pipeline_energy_demand.pkl",
                dst_path=tmp_dir
            )
            _pipeline = joblib.load(local_path)

        print(f"Pipeline loaded from MLflow run: {run_id} (version {version})")

    except Exception as e:
        print(f"WARNING: Could not load pipeline from MLflow ({e}). Falling back to local file.")
        _pipeline = joblib.load(settings.PIPELINE_PATH)

    return _pipeline


def get_loaded_version() -> str:
    return _loaded_version or "unknown"


def reload_models():
    """Forces a reload of both model and pipeline on the next request. Used after model transition."""
    global _model, _pipeline
    _model = None
    _pipeline = None
