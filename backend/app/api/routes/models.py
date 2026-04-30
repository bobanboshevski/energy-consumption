from fastapi import APIRouter

from app.core.activate_model import get_active_model_state, set_active_version
from app.core.mlflow_client import get_registered_models, get_experiment_runs, get_all_model_versions, \
    transition_model_version
from app.core.model_loader import reload_models, get_loaded_version

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/registry")
def registry():
    return get_registered_models()


@router.get("/experiments/{experiment_name}")
def experiments(experiment_name: str):
    return get_experiment_runs(experiment_name)


# @router.get("/transition")
# def transition(model_name: str, version: str, stage: str):
#     result = transition_model_version(model_name, version, stage)
#     # Force reload so next prediction request uses the newly promoted model
#     reload_models()
#     return result

@router.get("/versions/{model_name}")
def versions(model_name: str):
    """Returns ALL versions of a model with their metrics."""
    return get_all_model_versions(model_name)


@router.get("/experiments/{experiment_name}")
def experiments(experiment_name: str):
    return get_experiment_runs(experiment_name)


# todo: I also want to get the model version that is CURRENTLY USED!
# @router.get("/current/{model_name}")
# def current_version(model_name: str):
#     return get_current_model(model_name)



# TODO: WE SHOULD DISPLAY THE VERSIONS OF THE MODELS ON THE UI AS ENUMS, Since if we write a version that doesn't exists
# todo: i falls back the the latest one
@router.get("/active")
def active():
    """Returns which model version is currently active and loaded."""
    state = get_active_model_state()
    state["loaded_version"] = get_loaded_version()
    return state


@router.post("/activate")
def activate(version: str):
    """Sets a specific model version as active. Backend reloads the model on next request."""
    result = set_active_version(version)
    reload_models()
    return result

@router.post("/transition")
def transition(model_name: str, version: str, stage: str):
    """Promotes a model version to Staging, Production, or Archived."""
    result = transition_model_version(model_name, version, stage)
    reload_models()
    return result

# @router.post("/alias")
# def set_alias(model_name: str, alias: str, version: str):
#     result = set_model_alias(model_name, alias, version)
#     reload_models()
#     return result
#
#
# @router.delete("/alias")
# def remove_alias(model_name: str, alias: str):
#     result = delete_model_alias(model_name, alias)
#     reload_models()
#     return result
