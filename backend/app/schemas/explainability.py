"""
Pydantic response schemas for SHAP explainability endpoints.
These define the exact shape of the JSON returned to the frontend.
"""
from typing import Optional

from pydantic import BaseModel, field_validator, ConfigDict


class FeatureImportance(BaseModel):
    """Mean absolute SHAP per feature, aggregated across all timesteps."""
    energy_demand: float
    temp_max: float
    temp_min: float
    daylight_duration: float


class ExplanationRecord(BaseModel):
    """
    SHAP explanation for a single forecast date.

    shap_matrix:         Raw (window_size × n_features) SHAP values.
                         Rows = timesteps (30 past days), columns = features (4).
                         Used for heatmap rendering on the frontend.

    timestep_importance: Mean absolute SHAP per timestep, collapsed across features.
                         Length = window_size. Tells you which past day mattered most.

    feature_importance:  Mean absolute SHAP per feature, collapsed across timesteps.
                         Tells you which variable mattered most overall.

    base_value:          Average model output over the background dataset (scaled space).
                         baseline + sum(shap_matrix) ≈ model output for this prediction.
    """
    date: str
    predicted_demand: float
    base_value: float
    feature_importance: FeatureImportance
    timestep_importance: list[float]
    shap_matrix: list[list[float]]

    @field_validator("shap_matrix", mode="before")
    @classmethod
    def normalise_shap_matrix(cls, matrix: list) -> list[list[float]]:
        """
        Normalises every cell in the shap_matrix to a plain float.

        Keras GradientExplainer wraps single-output model predictions in an extra
        dimension, so cells arrive as [0.004327] instead of 0.004327. This validator
        handles both formats transparently so consumers always receive plain floats.

        Input:  list[list[float | list[float]]]   (either format, or mixed)
        Output: list[list[float]]                 (always plain floats)
        """
        result = []
        for row in matrix:
            normalised_row = []
            for cell in row:
                if isinstance(cell, (list, tuple)):
                    # Keras: cell = [0.004327] → extract first element
                    normalised_row.append(float(cell[0]) if cell else 0.0)
                else:
                    normalised_row.append(float(cell) if cell is not None else 0.0)
            result.append(normalised_row)
        return result


class ExplainabilityResponse(BaseModel):
    """Full SHAP artifact for one model variant."""
    model_variant: str
    shap_method: str
    version: str
    generated_at: str
    n_background_samples: int
    feature_names: list[str]
    window_size: int
    n_explanations: int
    explanations: list[ExplanationRecord]


class ExplainabilityUnavailableResponse(BaseModel):
    """Returned when SHAP artifacts don't exist for a requested model version."""
    available: bool = False
    model_variant: str
    version: str
    reason: str


class ShapNarrative(BaseModel):
    # extra="ignore" keeps us robust if the LLM adds/omits fields
    model_config = ConfigDict(extra="ignore")

    date: Optional[str] = None
    variant: Optional[str] = None
    headline: Optional[str] = None
    predicted_demand_gw: Optional[float] = None
    top_feature: Optional[str] = None
    top_feature_share_pct: Optional[float] = None
    most_influential_day: Optional[str] = None
    key_findings: list[str] = []
    summary: Optional[str] = None


class NarrativeResponse(BaseModel):
    variant: str
    date: str
    version: Optional[str] = None
    predicted_demand: float  # authoritative value from the SHAP artifact, NOT the LLM's rounded one
    narrative: ShapNarrative
