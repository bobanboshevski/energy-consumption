from pathlib import Path
import json
from threading import Lock  # todo: what is this?

_STATE_FILE = Path(__file__).resolve().parent.parent.parent / "active_model.json"
_lock = Lock()

# Supported model keys
MODEL_MULTIVARIATE = "multivariate"
MODEL_UNIVARIATE = "univariate"

_DEFAULT = {
    MODEL_MULTIVARIATE: {"active_version": "latest", "loaded_version": None},
    MODEL_UNIVARIATE: {"active_version": "latest", "loaded_version": None},
}


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


def get_active_version(model_key: str = MODEL_MULTIVARIATE) -> str:
    """Returns the configured active version for the given model key."""
    with _lock:
        return _load_state().get(model_key, {}).get("active_version", "latest")


def set_active_version(version: str, model_key: str = MODEL_MULTIVARIATE) -> dict:
    """
    Sets the active version. Does NOT update loaded_version —
    that only changes when the model is actually loaded into memory.
    """
    with _lock:
        state = _load_state()
        if model_key not in state:
            state[model_key] = {"active_version": "latest", "loaded_version": None}
        state[model_key]["active_version"] = version
        _save_state(state)
    return {"success": True, "model_key": model_key, "active_version": version}


def set_loaded_version(version: str, model_key: str = MODEL_MULTIVARIATE):
    """
    Called by model_loader after a model is successfully loaded into memory.
    Updates loaded_version in the JSON so the UI always reflects reality.
    """
    with _lock:
        state = _load_state()
        if model_key not in state:
            state[model_key] = {"active_version": "latest", "loaded_version": None}
        state[model_key]["loaded_version"] = version
        _save_state(state)


def get_full_state() -> dict:
    """Returns the full active model state for all models."""
    with _lock:
        return _load_state()
