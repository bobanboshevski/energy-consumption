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
# todo: i duplicated that code, which is not the best way to do it, but because of that it's not shared anymore
_SHARED_DIR = Path(__file__).resolve().parent.parent.parent.parent / "shared"
sys.path.insert(0, str(_SHARED_DIR))

# ── Multivariate model (model 1) ──────────────────────────────────────────────
_model = None
_pipeline = None
_loaded_version = None

# ── Univariate model (model 2) ────────────────────────────────────────────────
_univariate_model = None
_univariate_pipeline = None
_univariate_loaded_version = None


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


def _resolve_univariate_version() -> str:
    """Resolves the latest version of the univariate model."""
    try:
        _setup_mlflow()
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{settings.MLFLOW_UNIVARIATE_MODEL_NAME}'")
        if versions:
            return max(versions, key=lambda v: int(v.version)).version
    except Exception as e:
        print(f"WARNING: Could not resolve univariate version: {e}")

    return "1"


def _load_pipeline_from_mlflow(model_name: str, version: str, artifact_filename: str):
    """Downloads and loads a pipeline artifact from MLflow."""
    _setup_mlflow()
    client = mlflow.tracking.MlflowClient()
    model_version = client.get_model_version(model_name, version)
    run_id = model_version.run_id

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact_filename,
            dst_path=tmp_dir
        )
        return joblib.load(local_path)


# ── Multivariate loaders ──────────────────────────────────────────────────────
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


# def load_pipeline():
#     global _pipeline
#     if _pipeline is not None:
#         return _pipeline
#
#     version = _resolve_version()
#
#     try:
#         _setup_mlflow()
#         client = mlflow.tracking.MlflowClient()
#
#         model_version = client.get_model_version(settings.MLFLOW_MODEL_NAME, version)
#         run_id = model_version.run_id
#
#         with tempfile.TemporaryDirectory() as tmp_dir:
#             local_path = mlflow.artifacts.download_artifacts(
#                 run_id=run_id,
#                 artifact_path="pipeline_energy_demand.pkl",
#                 dst_path=tmp_dir
#             )
#             _pipeline = joblib.load(local_path)
#
#         print(f"Pipeline loaded from MLflow run: {run_id} (version {version})")
#
#     except Exception as e:
#         print(f"WARNING: Could not load pipeline from MLflow ({e}). Falling back to local file.")
#         _pipeline = joblib.load(settings.PIPELINE_PATH)
#
#     return _pipeline

def load_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    version = _resolve_version()

    try:
        _pipeline = _load_pipeline_from_mlflow(
            settings.MLFLOW_MODEL_NAME, version, "pipeline_energy_demand.pkl"
        )
        print(f"Multivariate pipeline loaded from MLflow (version {version})")
    except Exception as e:
        print(f"WARNING: Could not load multivariate pipeline from MLflow ({e}). Falling back to local.")
        _pipeline = joblib.load(settings.PIPELINE_PATH)

    return _pipeline


# ── Univariate loaders ────────────────────────────────────────────────────────
def load_univariate_model():
    global _univariate_model, _univariate_loaded_version
    if _univariate_model is not None:
        return _univariate_model

    version = _resolve_univariate_version()

    try:
        _setup_mlflow()
        model_uri = f"models:/{settings.MLFLOW_UNIVARIATE_MODEL_NAME}/{version}"
        _univariate_model = mlflow.tensorflow.load_model(model_uri)
        _univariate_loaded_version = version
        print(f"Univariate model loaded from MLflow: {model_uri}")
    except Exception as e:
        print(f"WARNING: Could not load univariate model from MLflow ({e}). Falling back to local.")
        _univariate_model = keras.models.load_model(settings.UNIVARIATE_MODEL_PATH)
        _univariate_loaded_version = "local"

    return _univariate_model


def load_univariate_pipeline():
    global _univariate_pipeline
    if _univariate_pipeline is not None:
        return _univariate_pipeline

    version = _resolve_univariate_version()

    try:
        _univariate_pipeline = _load_pipeline_from_mlflow(
            settings.MLFLOW_UNIVARIATE_MODEL_NAME, version, "pipeline_univariate.pkl"
        )
        print(f"Univariate pipeline loaded from MLflow (version {version})")
    except Exception as e:
        print(f"WARNING: Could not load univariate pipeline from MLflow ({e}). Falling back to local.")
        _univariate_pipeline = joblib.load(settings.UNIVARIATE_PIPELINE_PATH)

    return _univariate_pipeline


# ── Utilities ─────────────────────────────────────────────────────────────────
def get_loaded_version() -> str:
    return _loaded_version or "unknown"


def get_univariate_loaded_version() -> str:
    return _univariate_loaded_version or "unknown"


# def reload_models():
#     """Forces a reload of both model and pipeline on the next request. Used after model transition."""
#     global _model, _pipeline
#     _model = None
#     _pipeline = None

def reload_models():
    """Forces reload of all models on next request. Called after model transition."""
    global _model, _pipeline, _loaded_version
    global _univariate_model, _univariate_pipeline, _univariate_loaded_version
    _model = None
    _pipeline = None
    _loaded_version = None
    _univariate_model = None
    _univariate_pipeline = None
    _univariate_loaded_version = None
