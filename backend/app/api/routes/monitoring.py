from starlette.responses import HTMLResponse

from app.core.data_service import get_drift_report_html, get_gx_report_html
from app.serivces.monitoring_service import get_model_performance_over_time, get_drift_report_summary, \
    get_current_metrics, get_univariate_performance_over_time, get_univariate_metrics, get_gx_report_summary
from fastapi import APIRouter

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


# ── Multivariate ──────────────────────────────────────────────────────────────
@router.get("/performance")
def performance(window_days: int = 30):
    return get_model_performance_over_time(window_days)


@router.get("/metrics")
def metrics(window_days: int = 30):
    return get_current_metrics(window_days)


# ── Univariate ────────────────────────────────────────────────────────────────

@router.get("/univariate/performance")
def univariate_performance(window_days: int = 30):
    return get_univariate_performance_over_time(window_days)


@router.get("/univariate/metrics")
def univariate_metrics(window_days: int = 30):
    return get_univariate_metrics(window_days)


# ── Drift ─────────────────────────────────────────────────────────────────────
# summary
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


# ── Data structure report ─────────────────────────────────────────────────────────

# summary
@router.get("/gx")
def gx_summary():
    return get_gx_report_summary()


@router.get("/gx/report", response_class=HTMLResponse)
def gx_report():
    """Returns the full GX validation HTML report for embedding in an iframe."""
    html = get_gx_report_html()
    if html is None:
        return HTMLResponse("<p>GX report not available.</p>", status_code=404)
    return HTMLResponse(html)
