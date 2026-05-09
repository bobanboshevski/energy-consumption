import io
import time
import threading
import pandas as pd
import httpx
from pathlib import Path
from datetime import datetime, timedelta
from app.core.config import settings

_cache: dict = {
    "data": None,
    "loaded_at": None,
}
_lock = threading.Lock()


def _dagshub_url(file_path: str) -> str:
    """Builds the DagShub raw content URL for a file tracked in the repo."""
    return (
        f"https://dagshub.com/{settings.DAGSHUB_USERNAME}"
        f"/{settings.DAGSHUB_REPO}/raw/{settings.DAGSHUB_BRANCH}/{file_path}"
    )


def _download_from_dagshub(file_path: str) -> bytes:
    """Downloads a file from DagShub using HTTP Basic Auth."""
    url = _dagshub_url(file_path)
    print(f"Downloading data from DagShub: {url}")

    with httpx.Client(timeout=60.0) as client:
        response = client.get(
            url,
            auth=(settings.DAGSHUB_USERNAME, settings.MLFLOW_TRACKING_PASSWORD),
            follow_redirects=True,
        )
        response.raise_for_status()
        return response.content


def _is_cache_fresh() -> bool:
    """Returns True if cached data is less than CACHE_TTL_MINUTES old."""
    if _cache["data"] is None or _cache["loaded_at"] is None:
        return False
    age = datetime.utcnow() - _cache["loaded_at"]
    return age < timedelta(minutes=settings.CACHE_TTL_MINUTES)


def get_data(force_refresh: bool = False) -> pd.DataFrame:
    """
    Returns the preprocessed dataset.

    Priority:
    1. In-memory cache (if fresh)
    2. DagShub HTTP download
    3. Local file fallback

    The cache refreshes automatically every CACHE_TTL_MINUTES minutes.
    """
    with _lock:
        if not force_refresh and _is_cache_fresh():
            return _cache["data"].copy()

        df = None

        # Try DagShub first
        try:
            content = _download_from_dagshub(settings.DAGSHUB_DATA_PATH)
            df = pd.read_csv(io.BytesIO(content))
            print(f"Data loaded from DagShub ({len(df)} rows)")
        except Exception as e:
            print(f"WARNING: Could not download data from DagShub ({e}). Falling back to local file.")

        # Fall back to local file
        if df is None:
            local_path = Path(settings.DATA_PATH)
            if not local_path.exists():
                raise FileNotFoundError(
                    f"Data not available: DagShub download failed and local file not found at {local_path}"
                )
            df = pd.read_csv(local_path)
            print(f"Data loaded from local file ({len(df)} rows)")

        _cache["data"] = df
        _cache["loaded_at"] = datetime.utcnow()

        return df.copy()


def invalidate_cache():
    """Forces the next request to re-download data."""
    with _lock:
        _cache["data"] = None
        _cache["loaded_at"] = None
    print("Data cache invalidated.")


def get_drift_report_html() -> str | None:
    """
    Downloads the Evidently drift report HTML from DagShub.
    Falls back to local file if unavailable.
    """
    dagshub_path = f"machine-learning/reports/data_testing_report.html"  # todo: this can be add to config.py

    try:
        content = _download_from_dagshub(dagshub_path)
        return content.decode("utf-8")
    except Exception as e:
        print(f"WARNING: Could not download drift report from DagShub ({e}). Falling back to local.")

    local_path = Path(settings.DRIFT_REPORT_PATH)
    print(f"Using the local drift report at: {local_path}")
    if local_path.exists():
        return local_path.read_text()

    return None


def get_gx_report_html() -> str | None:
    """
        Downloads the Great Expectations validation report HTML from DagShub.
        Falls back to local file if unavailable.
        """
    try:
        content = _download_from_dagshub(settings.DAGSHUB_GX_REPORT_PATH)
        return content.decode("utf-8")
    except Exception as e:
        print(f"WARNING: Could not download GX report from DagShub ({e}). Falling back to local.")

    local_path = Path(settings.GX_REPORT_PATH)
    if local_path.exists():
        return local_path.read_text()

    return None
