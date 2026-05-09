from datetime import date, timedelta
from fastapi import APIRouter, HTTPException, Query

# from app.services.univariate_prediction_service import predict_for_date, predict_range
from app.serivces.univariate_prediction_service import predict_for_date, predict_range
from app.core.config import settings

router = APIRouter(prefix="/univariate", tags=["univariate"])


@router.get("/predict")
def predict(target_date: date = Query(..., description="Date to predict (YYYY-MM-DD)")):
    """
    Predicts energy demand for a specific future date.
    Uses the univariate LSTM model (energy demand only — no weather data required).
    Max horizon: 365 days from today.
    """
    try:
        return predict_for_date(target_date)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/range")
def predict_range_endpoint(
        start_date: date = Query(..., description="Start date (YYYY-MM-DD)"),
        end_date: date = Query(..., description="End date (YYYY-MM-DD)"),
):
    """
    Predicts energy demand for every day between start_date and end_date.
    Useful for rendering a long-range forecast chart.
    Max horizon: 365 days from today.
    """
    try:
        return predict_range(start_date, end_date)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/info")
def model_info():
    """Returns metadata about the univariate model and its capabilities."""
    today = date.today()
    return {
        "model": "univariate_lstm",
        "description": "Energy demand prediction using historical demand patterns only",
        "features": ["energy_demand"],
        "window_size": settings.UNIVARIATE_WINDOW_SIZE,
        "max_horizon_days": settings.UNIVARIATE_MAX_HORIZON_DAYS,
        "max_date": str(today + timedelta(days=settings.UNIVARIATE_MAX_HORIZON_DAYS)),
        "note": "Accuracy decreases for dates further in the future",
    }
