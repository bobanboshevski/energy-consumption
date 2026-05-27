from fastapi import APIRouter

from app.core.data_service import invalidate_cache
from app.services.prediction_service import get_forecast_data, get_historical_data

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("/forecast")
def forecast():
    return get_forecast_data()


@router.get("/historical")
def historical(days: int = 90):
    return get_historical_data(days)


@router.post("/refresh")
def refresh_data():
    """Forces a re-download of the latest data from DagShub."""
    invalidate_cache()
    return {"success": True, "message": "Data cache cleared. Next request will re-download from DagShub."}
