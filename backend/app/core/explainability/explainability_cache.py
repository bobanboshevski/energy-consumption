import json
import os
import tempfile

import mlflow

from app.core.config import settings
from app.core.exceptions.exceptions import ModelVersionNotFound, ExplainabilityNotAvailable, MLflowUnavailable
from app.core.mlflow_client import get_run_id_for_version

"""
In-memory cache for SHAP explanation artifacts.

SHAP artifacts are version-specific and immutable — once generated during
training, they never change for a given model version. The cache accumulates
entries as different versions are requested; no invalidation is ever needed.

Cache key: (version, variant)  e.g. ("32", "keras")
Cache value: the parsed JSON artifact dict
"""

# variant name → artifact filename in MLflow
_ARTIFACT_FILENAMES: dict[str, str] = {
    "keras": "shap_explanations_keras.json",
    "onnx": "shap_explanations_onnx.json",
    "onnx_quantized": "shap_explanations_onnx_quantized.json",
}

# In-memory store: (version, variant) → artifact dict
_cache: dict[tuple[str, str], dict] = {}


def _setup_mlflow():
    os.environ["MLFLOW_TRACKING_USERNAME"] = settings.MLFLOW_TRACKING_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = settings.MLFLOW_TRACKING_PASSWORD
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)


def _download_artifact(run_id: str, variant: str) -> dict:
    """
    Downloads the SHAP JSON artifact from MLflow for the given run and variant.

    Raises ExplainabilityNotAvailable if the artifact file does not exist —
    this is the expected case for models trained before SHAP was added.
    Raises MLflowUnavailable for network/auth failures.
    """
    filename = _ARTIFACT_FILENAMES[variant]
    try:
        _setup_mlflow()
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path=filename,
                dst_path=tmp_dir,
            )
            with open(local_path) as f:
                data = json.load(f)

        n = data.get("n_explanations", 0)
        print(f"[SHAP CACHE] ✓  Downloaded {filename} — {n} explanations")
        return data

    except Exception as e:
        msg = str(e)
        if any(phrase in msg.lower() for phrase in ["no such file", "path is correct", "does not exist", "not found"]):
            # File doesn't exist — expected for older model versions
            print(
                f"[SHAP CACHE] —  {filename} not found for run_id={run_id[:8]}... (model trained before explainability was added)"
            )
            raise ExplainabilityNotAvailable(version="unknown",
                                             variant=variant) from e  # todo: actual model version can be passed!
        # Network or auth failure
        print(f"[SHAP CACHE] ✗  Download failed for {filename}: {msg}")
        raise MLflowUnavailable(msg) from e


def get_shap_artifact(
        version: str,
        variant: str,
        model_name: str = None,
) -> dict:
    """
        Returns the SHAP artifact for a given model version and variant.
        Uses in-memory cache — downloads from MLflow only on first request
        for a given (version, variant) pair.

        Args:
            version:    Model version number string e.g. "32"
            variant:    One of "keras", "onnx", "onnx_quantized"
            model_name: MLflow registered model name (defaults to multivariate)

        Returns:
            Parsed SHAP artifact dict.

        Raises:
            ModelVersionNotFound:       Version doesn't exist in the registry.
            ExplainabilityNotAvailable: SHAP file not found for this version.
            MLflowUnavailable:          MLflow tracking server unreachable.
    """

    if model_name is None:
        model_name = settings.MLFLOW_MODEL_NAME

    cache_key = (version, variant)

    if cache_key in _cache:
        print(f"[SHAP CACHE] ✓  HIT (v{version}, {variant})")
        return _cache[cache_key]

    print(f"[SHAP CACHE] →  MISS (v{version}, {variant}) — fetching from MLflow...")

    # Delegate the MLflow call to mlflow_client — it already knows how to talk to MLflow
    run_id = get_run_id_for_version(model_name, version)
    if run_id is None:
        raise ModelVersionNotFound(model_name, version)

    artifact = _download_artifact(run_id, variant)
    artifact["version"] = version

    _cache[cache_key] = artifact
    print(f"[SHAP CACHE] ✓  Cached (v{version}, {variant})")
    return artifact


def clear_cache():
    """Clears all cached SHAP artifacts. Useful for testing."""
    _cache.clear()
    print("[SHAP CACHE] All entries cleared.")
