"""
Domain-specific exceptions for the explainability and model serving layer.
These are caught at the route layer and mapped to appropriate HTTP status codes.
"""


class ModelVersionNotFound(Exception):
    """
    Raised when the requested model version does not exist in the MLflow registry.
    Maps to HTTP 404.
    """

    def __init__(self, model_name: str, version: str):
        self.model_name = model_name
        self.version = version
        super().__init__(
            f"Model '{model_name}' version '{version}' does not exist in the registry."
        )


class ExplainabilityNotAvailable(Exception):
    """
    Raised when SHAP artifacts have not been generated for a model version.
    This is expected for versions trained before explainability was added.
    Maps to HTTP 404.
    """

    def __init__(self, version: str, variant: str):
        self.version = version
        self.variant = variant
        super().__init__(
            f"SHAP explanations ({variant}) are not available for model version {version}. "
            f"This version was trained before explainability was added to the pipeline. "
            f"Re-train the model to generate SHAP artifacts."
        )


class ExplanationDateNotFound(Exception):
    """
    Raised when the requested forecast date is not present in the SHAP artifact.
    Maps to HTTP 404.
    """

    def __init__(self, date: str, version: str, available_dates: list[str]):
        self.date = date
        self.version = version
        self.available_dates = available_dates
        super().__init__(
            f"No SHAP explanation found for date '{date}' in version {version}. "
            f"Available dates: {available_dates}"
        )


class MLflowUnavailable(Exception):
    """
    Raised when the MLflow tracking server is unreachable.
    Maps to HTTP 503.
    """

    def __init__(self, detail: str):
        super().__init__(f"MLflow is unreachable: {detail}")


class NarrativeUnavailable(Exception):
    """Raised when the LLM narrative cannot be generated (upstream LLM failure
    or malformed response). Mapped to HTTP 502."""
    pass
