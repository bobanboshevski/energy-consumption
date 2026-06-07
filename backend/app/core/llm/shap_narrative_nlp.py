import json

import requests

from app.core.exceptions.exceptions import NarrativeUnavailable
from app.core.llm.llm_client import chat_completion
from app.core.llm.prompts import SHAP_NARRATIVE_SYSTEM_PROMPT

"""
Builds the compact LLM payload from a SHAP artifact and turns the LLM response
into a parsed narrative dict.

Two responsibilities, both small:
  build_llm_payload  — pure transformation of artifact → compact payload
  get_shap_narrative — payload → LLM → parsed JSON dict
"""

_SHAP_METHOD_LABELS = {
    "keras": "GradientExplainer (TensorFlow autodiff)",
    "onnx": "KernelExplainer (model-agnostic perturbation)",
    "onnx_quantized": "KernelExplainer on INT8 quantized model",
}

_USER_PREAMBLE = "Analyze this SHAP data and return the JSON summary.\n\n"


def _find_record(artifact: dict, date: str) -> dict:
    """Returns the explanation record for the given date, or raises ValueError."""
    for record in artifact.get("explanations", []):
        if record.get("date") == date:
            return record
    raise ValueError(f"No explanation record for date {date} in artifact.")


def build_llm_payload(artifact: dict, date: str, variant_name: str) -> dict:
    """
    Transforms a SHAP artifact into the compact payload the LLM expects.

    - feature_importance_sorted: features sorted desc by importance, with share_pct
      (importance / total * 100) and net_direction (sign of the feature's SHAP column sum).
    - top_3_influential_timesteps: top 3 timesteps by |importance|, labelled D-{window-index}.
    """
    record = _find_record(artifact, date)

    feature_names: list[str] = artifact["feature_names"]
    window_size: int = artifact.get("window_size", len(record["timestep_importance"]))
    feature_importance: dict = record["feature_importance"]
    shap_matrix: list = record["shap_matrix"]
    timestep_importance: list = record["timestep_importance"]

    # Net direction per feature = sign of the sum of its column across all timesteps
    column_sums = [0.0] * len(feature_names)
    for row in shap_matrix:
        for col, value in enumerate(row):
            column_sums[col] += value
    name_to_index = {name: i for i, name in enumerate(feature_names)}

    total_importance = sum(feature_importance.values()) or 1e-12
    feature_importance_sorted = []
    for feature, importance in sorted(feature_importance.items(), key=lambda kv: kv[1], reverse=True):
        idx = name_to_index[feature]
        direction = "pushes_prediction_up" if column_sums[idx] > 0 else "pushes_prediction_down"
        feature_importance_sorted.append({
            "feature": feature,
            "importance": round(importance, 6),
            "share_pct": round(importance / total_importance * 100, 1),
            "net_direction": direction,
        })

    # Top 3 timesteps by absolute importance; index 0 = oldest = D-{window_size}
    ranked = sorted(enumerate(timestep_importance), key=lambda iv: abs(iv[1]), reverse=True)
    top_3_timesteps = []
    for index, importance in ranked[:3]:
        days_ago = window_size - index
        top_3_timesteps.append({
            "label": f"D-{days_ago}",
            "days_ago": days_ago,
            "importance": round(importance, 6),
        })

    return {
        "selected_date": date,
        "variant": {
            "name": variant_name,
            "shap_method": _SHAP_METHOD_LABELS.get(variant_name, variant_name),
            "predicted_demand_gw": record["predicted_demand"],
            "base_value": record["base_value"],
            "feature_importance_sorted": feature_importance_sorted,
            "top_3_influential_timesteps": top_3_timesteps,
        },
    }


def get_shap_narrative(payload: dict) -> dict:
    """
    Sends the payload to the LLM and returns the parsed narrative dict.
    Ignores model metadata (reasoning_content, usage, etc.) — only message.content matters.

    Raises:
        NarrativeUnavailable — on HTTP failure, unexpected response shape, or invalid JSON.
    """
    # Compact JSON (no spaces) to match the tested request format
    user_content = _USER_PREAMBLE + json.dumps(payload, separators=(",", ":"))
    messages = [
        {"role": "system", "content": SHAP_NARRATIVE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        response = chat_completion(messages)
    except requests.RequestException as e:
        raise NarrativeUnavailable(f"LLM request failed: {e}") from e

    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise NarrativeUnavailable(f"Unexpected LLM response shape: {e}") from e

    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError) as e:
        raise NarrativeUnavailable(f"LLM did not return valid JSON: {e}") from e
