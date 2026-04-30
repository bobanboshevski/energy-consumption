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
    MLFLOW_MODEL_NAME: str = "energy_demand_model"
    MLFLOW_MODEL_ALIAS: str = "champion"  # modern MLflow uses aliases not stages

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
    DRIFT_REPORT_PATH: str = str(_ML_DIR / "reports/data_testing_report.html")

    # Model config
    WINDOW_SIZE: int = 30
    TARGET_COL: str = "energy_demand"
    FEATURE_COLS: list = ["temp_max", "temp_min", "daylight_duration"]

    # Second model — uncomment when classifier is ready
    # CLASSIFIER_PATH: str = str(_ML_DIR / "models/classifier_energy_demand.pkl")
    # LABEL_ENCODER_PATH: str = str(_ML_DIR / "models/label_encoder.pkl")
    # THRESHOLDS_PATH: str = str(_ML_DIR / "models/demand_thresholds.pkl")

    class Config:
        env_file = ".env"


settings = Settings()
