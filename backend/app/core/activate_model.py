from pathlib import Path
import json
from threading import Lock # todo: what is this?

_STATE_FILE = Path(__file__).resolve().parent.parent.parent / "active_model.json"
_lock = Lock()

_DEFAULT = {
    "model_name": "energy_demand_model",
    "active_version": "latest",
}


def _load_state() -> dict:
    if _STATE_FILE.exists():
        with open(_STATE_FILE) as f:
            return json.load(f)
    return _DEFAULT.copy()


def _save_state(state: dict):
    with open(_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_active_version() -> str:
    with _lock:
        return _load_state().get("active_version", "latest") # todo: check this


def set_active_version(version: str) -> dict:
    with _lock:
        state = _load_state()
        state["active_version"] = version
        _save_state(state)
    return {"success": True, "active_version": version}


def get_active_model_state() -> dict:
    with _lock:
        return _load_state()
