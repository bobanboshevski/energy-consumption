from typing import Optional

from fastapi import APIRouter, Query, HTTPException

from app.core.exceptions.exceptions import ModelVersionNotFound, ExplainabilityNotAvailable, ExplanationDateNotFound, \
    MLflowUnavailable
from app.schemas.explainability import ExplainabilityResponse
from app.services.explainability_service import get_explanations

"""
SHAP model explainability endpoints — one per model variant.

All three endpoints share the same query parameters and response structure.
They are kept as separate functions (rather than a single parameterised endpoint)
so that each variant is explicit and independently documented in the API schema.

Endpoints:
  GET /explainability/keras
  GET /explainability/onnx
  GET /explainability/onnx_quantized

Query parameters (all optional):
  version  Model version number e.g. "32". Defaults to the currently active version.
  date     ISO date (YYYY-MM-DD). If provided, returns only that date's explanation.
           If omitted, returns all forecast dates.
"""

router = APIRouter(prefix="/explainability", tags=["explainability"])

# ── Shared query parameter descriptions ──────────────────────────────────────

_VERSION_QUERY = Query(
    default=None,
    description="Model version number (e.g. '32'). Defaults to the currently active version.",
)
_DATE_QUERY = Query(
    default=None,
    description="ISO date (YYYY-MM-DD). Returns only that date's explanation. Omit for all dates.",
)


def _handle_explainability_request(variant: str, version: Optional[str], date: Optional[str]) -> dict:
    """
    Shared handler — calls the service and maps domain exceptions to HTTP responses.
    All three route functions delegate here to avoid code duplication.
    """
    try:
        return get_explanations(variant=variant, version=version, date=date)

    except ModelVersionNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))

    except ExplainabilityNotAvailable as e:
        raise HTTPException(status_code=404, detail=str(e))

    except ExplanationDateNotFound as e:
        raise HTTPException(
            status_code=404,
            detail={
                "message": str(e),
                "available_dates": e.available_dates,
            },
        )

    except MLflowUnavailable as e:
        raise HTTPException(
            status_code=503,
            detail=f"MLflow tracking server is unavailable. {e}",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error retrieving SHAP explanations: {e}",
        )


# ── Variant: Keras ─────────────────────────────────────────────────────────────

@router.get(
    "/keras",
    response_model=ExplainabilityResponse,
    summary="Keras model SHAP explanations",
    description=(
            "Returns SHAP explanations computed with GradientExplainer on the Keras model. "
            "GradientExplainer uses TensorFlow automatic differentiation — "
            "fastest of the three variants."
    ),
)
def keras_explanations(
        version: Optional[str] = _VERSION_QUERY,
        date: Optional[str] = _DATE_QUERY,
) -> dict:
    return _handle_explainability_request("keras", version, date)


# ── Variant: ONNX base ─────────────────────────────────────────────────────────

@router.get(
    "/onnx",
    response_model=ExplainabilityResponse,
    summary="ONNX model SHAP explanations",
    description=(
            "Returns SHAP explanations computed with KernelExplainer on the base ONNX model. "
            "KernelExplainer is model-agnostic — required because ONNX Runtime "
            "does not expose gradients."
    ),
)
def onnx_explanations(
        version: Optional[str] = _VERSION_QUERY,
        date: Optional[str] = _DATE_QUERY,
) -> dict:
    return _handle_explainability_request("onnx", version, date)


# ── Variant: ONNX quantized ────────────────────────────────────────────────────

@router.get(
    "/onnx_quantized",
    response_model=ExplainabilityResponse,
    summary="ONNX INT8 quantized model SHAP explanations",
    description=(
            "Returns SHAP explanations computed with KernelExplainer on the INT8 quantized ONNX model. "
            "Comparing these values against the base ONNX and Keras variants shows "
            "whether quantization shifted the model's feature attribution."
    ),
)
def onnx_quantized_explanations(
        version: Optional[str] = _VERSION_QUERY,
        date: Optional[str] = _DATE_QUERY,
) -> dict:
    return _handle_explainability_request("onnx_quantized", version, date)
