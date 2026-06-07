from typing import Optional

from app.core.activate_model import MODEL_MULTIVARIATE, get_active_version
from app.core.config import settings
from app.core.exceptions.exceptions import ExplanationDateNotFound
from app.core.explainability.explainability_cache import get_shap_artifact
from app.core.llm.shap_narrative_nlp import build_llm_payload, get_shap_narrative
from app.core.model_loader import resolve_version

"""
Service layer for SHAP model explainability.

Responsibilities:
  - Resolve the effective model version (use active if none provided)
  - Delegate to the cache layer
  - Filter by date if requested
  - Raise typed domain exceptions — never HTTP exceptions

The route layer is the only place that maps exceptions to HTTP status codes.
"""

# Valid variant names and their display labels
VALID_VARIANTS = {"keras", "onnx", "onnx_quantized"}

# Narratives are deterministic for a given (version, variant, date) at temperature 0.1
# and the underlying artifact is immutable, so we can cache indefinitely.
_narrative_cache: dict[tuple, dict] = {}


def _resolve_version(version: Optional[str]) -> str:
    """
    Returns the version to use for explainability lookup.

    If version is None or "latest", resolves the currently active version
    from active_model.json — the same version displayed on the landing page
    and in the admin dashboard.
    """
    if version is None:
        return get_active_version(MODEL_MULTIVARIATE)
    return version


def get_explanations(
        variant: str,
        version: Optional[str] = None,
        date: Optional[str] = None,
) -> dict:
    """
    Returns SHAP explanations for a given model variant and version.

    Args:
        variant:  "keras", "onnx", or "onnx_quantized"
        version:  Model version number string, or None to use active version.
        date:     ISO date string (YYYY-MM-DD) to filter to a single explanation.
                  If None, returns all forecast dates.

    Returns:
        SHAP artifact dict, optionally filtered to a single date.

    Raises:
        ModelVersionNotFound:       Version doesn't exist.
        ExplainabilityNotAvailable: SHAP artifact missing for this version.
        ExplanationDateNotFound:    Requested date not in the artifact.
        MLflowUnavailable:          MLflow unreachable.
    """
    effective_version = _resolve_version(version)

    artifact = get_shap_artifact(
        version=effective_version,
        variant=variant,
        model_name=settings.MLFLOW_MODEL_NAME,
    )

    if date is None:
        return artifact

    # Filter to the requested date
    explanations = artifact.get("explanations", [])
    matched = [e for e in explanations if e["date"] == date]

    if not matched:
        available_dates = [e["date"] for e in explanations]
        raise ExplanationDateNotFound(
            date=date,
            version=effective_version,
            available_dates=available_dates,
        )

    return {
        **artifact,
        "explanations": matched,
        "n_explanations": len(matched),
    }


def get_narrative(variant: str, date: str, version: str | None = None) -> dict:
    """
    Returns an LLM-generated narrative for one (variant, date).

    Reuses get_explanations() for version resolution, artifact fetch, and date
    validation (so all ModelVersionNotFound / ExplainabilityNotAvailable /
    ExplanationDateNotFound / MLflowUnavailable exceptions propagate unchanged).
    """

    # Single-date filtered artifact — raises the appropriate domain exception if missing
    explanation = get_explanations(variant=variant, version=version, date=date)
    resolved_version = explanation.get("version")

    cache_key = (resolved_version, variant, date)
    if cache_key in _narrative_cache:
        return _narrative_cache[cache_key]

    payload = build_llm_payload(explanation, date, variant)
    narrative = get_shap_narrative(payload)

    result = {
        "variant": variant,
        "date": date,
        "version": resolved_version,
        # authoritative GW value from the artifact (LLM value is display-only/rounded); WE USE THE MOST ACCURATE VALUE
        "predicted_demand": payload["variant"]["predicted_demand_gw"],
        "narrative": narrative,
    }
    _narrative_cache[cache_key] = result
    return result
