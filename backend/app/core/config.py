from pydantic_settings import BaseSettings
from pathlib import Path

# todo: But this will work only if the backend is deployed with the /machine-learning directory
# todo: isnt there better pattern to do this? Like to get data from Dagshub - the data that the model was trained on?
# Resolve paths relative to this file's location
_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent  # backend/
_PROJECT_ROOT = _BACKEND_DIR.parent  # IIS projekt energija/
_ML_DIR = _PROJECT_ROOT / "machine-learning"  # machine-learning/


class Settings(BaseSettings):
    # todo: what about the more compressed version of the models?

    # MLflow
    MLFLOW_TRACKING_URI: str = "https://dagshub.com/bobanboshevski/energy-consumption.mlflow"
    MLFLOW_TRACKING_USERNAME: str = ""
    MLFLOW_TRACKING_PASSWORD: str = ""

    # Multivariate model (model 1 — weather + energy, 16 days ahead)
    MLFLOW_MODEL_NAME: str = "energy_demand_model"
    MLFLOW_MODEL_ALIAS: str = "champion"  # todo: this is not used

    # Univariate model (model 2 — energy only, up to 365 days ahead)
    MLFLOW_UNIVARIATE_MODEL_NAME: str = "energy_demand_univariate_model"
    UNIVARIATE_WINDOW_SIZE: int = 60
    UNIVARIATE_MAX_HORIZON_DAYS: int = 365

    # DagShub repo info — used to download data via HTTP API
    DAGSHUB_USERNAME: str = "bobanboshevski"
    DAGSHUB_REPO: str = "energy-consumption"
    DAGSHUB_BRANCH: str = "main"
    DAGSHUB_DATA_PATH: str = "machine-learning/data/preprocessed/final_dataset.csv"
    # todo: we should do the same for the reports!
    # todo: Reports: same as data — should be stored somewhere accessible

    # Cache TTL — how long to keep data in memory before re-downloading (minutes)
    CACHE_TTL_MINUTES: int = 60

    # Local fallback paths — used if DagShub is unreachable (development / offline)
    DATA_PATH: str = str(_ML_DIR / "data/preprocessed/final_dataset.csv")
    REFERENCE_PATH: str = str(_ML_DIR / "data/reference/final_dataset.csv")
    MODEL_PATH: str = str(_ML_DIR / "models/model_energy_demand.keras")
    PIPELINE_PATH: str = str(_ML_DIR / "models/pipeline_energy_demand.pkl")
    UNIVARIATE_MODEL_PATH: str = str(_ML_DIR / "models/model_energy_demand_univariate.keras")
    UNIVARIATE_PIPELINE_PATH: str = str(_ML_DIR / "models/pipeline_univariate.pkl")
    DRIFT_REPORT_PATH: str = str(_ML_DIR / "reports/data_testing_report.html")

    GX_REPORT_PATH: str = str(_ML_DIR / "reports/gx_validation_report.html")
    DAGSHUB_GX_REPORT_PATH: str = "machine-learning/reports/gx_validation_report.html"

    # Model config
    WINDOW_SIZE: int = 30
    TARGET_COL: str = "energy_demand"
    FEATURE_COLS: list = ["temp_max", "temp_min", "daylight_duration"]

    DEMAND_LOW_THRESHOLD: float = 1.2
    DEMAND_HIGH_THRESHOLD: float = 1.6

    # LLM narrative (SHAP summaries)
    LLM_ENDPOINT_URL: str = "https://isl-llm-infer.grega.xyz/v1/chat/completions"
    LLM_API_KEY: str = ""
    LLM_MODEL_NAME: str = "qwen3.6-27b"

    class Config:
        env_file = ".env"


settings = Settings()
