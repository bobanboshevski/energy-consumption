from starlette.responses import HTMLResponse

from app.core.data_service import get_drift_report_html
from app.serivces.monitoring_service import get_model_performance_over_time, get_drift_report_summary, \
    get_current_metrics
from fastapi import APIRouter

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/performance")
def performance(window_days: int = 30):
    return get_model_performance_over_time(window_days)


@router.get("/metrics")
def metrics():
    return get_current_metrics()

# todo: this is not working
@router.get("/drift")
def drift():
    return get_drift_report_summary()

@router.get("/drift/report", response_class=HTMLResponse)
def drift_report():
    """Returns the full Evidently HTML report for embedding in an iframe."""
    html = get_drift_report_html()
    if html is None:
        return HTMLResponse("<p>Report not available.</p>", status_code=404)
    return HTMLResponse(html)