from fastapi import APIRouter

from app.core.activate_model import set_active_version, get_active_version, MODEL_MULTIVARIATE, \
    MODEL_UNIVARIATE, get_full_state
from app.core.config import settings
from app.core.mlflow_client import get_registered_models, get_experiment_runs, get_all_model_versions, \
    transition_model_version
from app.core.model_loader import reload_models  # get_loaded_version, get_univariate_loaded_version

router = APIRouter(prefix="/models", tags=["models"])


# ── Registry ──────────────────────────────────────────────────────────────────
@router.get("/registry")
def registry():
    return get_registered_models()


@router.get("/experiments/{experiment_name}")
def experiments(experiment_name: str):
    return get_experiment_runs(experiment_name)


@router.get("/versions/{model_name}")
def versions(model_name: str):
    """Returns ALL versions of a model with their metrics."""
    return get_all_model_versions(model_name)


# todo: I also want to get the model version that is CURRENTLY USED!
# @router.get("/current/{model_name}")
# def current_version(model_name: str):
#     return get_current_model(model_name)


# ── Active model state ────────────────────────────────────────────────────────

# TODO: WE SHOULD DISPLAY THE VERSIONS OF THE MODELS ON THE UI AS ENUMS, Since if we write a version that doesn't exists
# todo: it falls back the the latest one

@router.post("/activate")
def activate(version: str, model_key: str = MODEL_MULTIVARIATE):
    """
    Sets the active version for a model.
    model_key: 'multivariate' or 'univariate'
    version: version number string or 'latest'
    """
    if model_key not in (MODEL_MULTIVARIATE, MODEL_UNIVARIATE):
        from fastapi import HTTPException
        raise HTTPException(status_code=400,
                            detail=f"Invalid model_key. Must be '{MODEL_MULTIVARIATE}' or '{MODEL_UNIVARIATE}'")

    result = set_active_version(version, model_key)
    reload_models()
    return result


@router.get("/active")
def active():
    """
    Returns active and loaded version for both models, read directly from active_model.json.
    active_version  — configured version (changes immediately on /activate)
    loaded_version  — version in memory (updates on next prediction request)
    """
    state = get_full_state()

    def model_info(model_key: str, model_name: str) -> dict:
        entry = state.get(model_key, {})
        active_v = entry.get("active_version", "latest")
        loaded_v = entry.get("loaded_version")
        return {
            "active_version": active_v,
            "loaded_version": loaded_v,
            "is_loaded": loaded_v is not None,
            "model_name": model_name,
        }

    return {
        "multivariate": model_info(MODEL_MULTIVARIATE, settings.MLFLOW_MODEL_NAME),
        "univariate": model_info(MODEL_UNIVARIATE, settings.MLFLOW_UNIVARIATE_MODEL_NAME),
    }


# todo: this is not used i think
@router.post("/transition")
def transition(model_name: str, version: str, stage: str):
    """Promotes a model version to Staging, Production, or Archived."""
    result = transition_model_version(model_name, version, stage)
    reload_models()
    return result
