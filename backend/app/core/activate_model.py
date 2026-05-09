from pathlib import Path
import json
from threading import Lock  # todo: what is this?

_STATE_FILE = Path(__file__).resolve().parent.parent.parent / "active_model.json"
_lock = Lock()

# Supported model keys
MODEL_MULTIVARIATE = "multivariate"
MODEL_UNIVARIATE = "univariate"

_DEFAULT = {
    MODEL_MULTIVARIATE: {"active_version": "latest"},
    MODEL_UNIVARIATE: {"active_version": "latest"},
}


# _DEFAULT = {
#     "model_name": "energy_demand_model",
#     "active_version": "latest",
# }


# def _load_state() -> dict:
#     if _STATE_FILE.exists():
#         with open(_STATE_FILE) as f:
#             return json.load(f)
#     return _DEFAULT.copy()


def _load_state() -> dict:
    if _STATE_FILE.exists():
        with open(_STATE_FILE) as f:
            data = json.load(f)
            # todo: this is for the old data format
            # Migrate old single-model format if needed
            if "active_version" in data:
                return _DEFAULT.copy()
            return data
    return _DEFAULT.copy()


def _save_state(state: dict):
    with open(_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# def get_active_version() -> str:
#     with _lock:
#         return _load_state().get("active_version", "latest")  # todo: check this

def get_active_version(model_key: str = MODEL_MULTIVARIATE) -> str:
    """Returns the configured active version for the given model key."""
    with _lock:
        return _load_state().get(model_key, {}).get("active_version", "latest")


# def set_active_version(version: str) -> dict:
#     with _lock:
#         state = _load_state()
#         state["active_version"] = version
#         _save_state(state)
#     return {"success": True, "active_version": version}

def set_active_version(version: str, model_key: str = MODEL_MULTIVARIATE) -> dict:
    """Sets the active version for the given model key."""
    with _lock:
        state = _load_state()
        if model_key not in state:
            state[model_key] = {}
        state[model_key]["active_version"] = version
        _save_state(state)
    return {"success": True, "model_key": model_key, "active_version": version}


def get_full_state() -> dict:
    """Returns the full active model state for all models."""
    with _lock:
        return _load_state()


# def get_active_model_state() -> dict:
#     with _lock:
#         return _load_state()
