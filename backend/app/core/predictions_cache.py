"""
Fetches cached forecast predictions from MLflow artifacts.

The training pipeline generates forecast_predictions.json after each training run
and logs it as an MLflow artifact. The backend fetches this artifact and serves
it directly, avoiding re-running the model on every request.

Cache validity is determined by comparing the forecast dates in the artifact
against the current forecast dates in the live dataset. If they match → serve cache.
If new dates have appeared (daily data update) → fall back to live inference.

Tier 1: MLflow artifact (generated at training time, version-specific)
Tier 2: Local file cache (written after live inference, reused across requests)

Priority on each request:
  1. In-memory (already fetched this session)
  2. MLflow artifact (if version matches)
  3. Local file cache (if dates match)
  4. None → caller runs live inference and calls save_live_cache()
"""

import json
import tempfile
import os
import mlflow
from datetime import datetime
from typing import Optional
from pathlib import Path

from app.core.config import settings
from app.core.activate_model import get_active_version, MODEL_MULTIVARIATE

_ARTIFACT_FILENAME = "forecast_predictions.json"
_LOCAL_CACHE_PATH = Path(__file__).resolve().parent.parent.parent / "forecast_cache.json"

_cached_artifact: Optional[dict] = None
_cached_for_version: Optional[str] = None


def _setup_mlflow():
    os.environ["MLFLOW_TRACKING_USERNAME"] = settings.MLFLOW_TRACKING_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = settings.MLFLOW_TRACKING_PASSWORD
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)


def _resolve_run_id(version: str) -> Optional[str]:
    """Gets the MLflow run_id for the given model version."""
    try:
        _setup_mlflow()
        client = mlflow.tracking.MlflowClient()
        mv = client.get_model_version(settings.MLFLOW_MODEL_NAME, version)
        return mv.run_id
    except Exception as e:
        print(f"[CACHE] ⚠  Could not resolve run_id for version {version}: {e}")
        return None


def _download_artifact(run_id: str) -> Optional[dict]:
    """Downloads forecast_predictions.json from the MLflow run."""
    try:
        _setup_mlflow()
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path=_ARTIFACT_FILENAME,
                dst_path=tmp_dir,
            )
            with open(local_path) as f:
                data = json.load(f)
        n = len(data.get("predictions", []))
        generated_at = data.get("generated_at", "unknown")
        print(f"[CACHE] ✓  MLflow artifact downloaded — {n} predictions (generated {generated_at})")
        return data

    except Exception as e:
        msg = str(e)
        if "please ensure that the path is correct" in msg or "No such file" in msg:
            # Expected for model versions trained before forecast artifact generation was added
            print(
                f"[CACHE] —  No forecast artifact for this model version (run_id={run_id[:8]}...) — falling back to live inference")
        else:
            # Unexpected: network issue, auth failure, etc.
            print(f"[CACHE] ⚠  MLflow artifact download error (run_id={run_id[:8]}...): {e}")
        return None


# ── Local file cache ──────────────────────────────────────────────────────────

def _read_local_cache() -> Optional[dict]:
    if not _LOCAL_CACHE_PATH.exists():
        print(f"[CACHE] —  No local cache file at {_LOCAL_CACHE_PATH}")
        return None
    try:
        with open(_LOCAL_CACHE_PATH) as f:
            data = json.load(f)
        n = len(data.get("predictions", []))
        generated_at = data.get("generated_at", "unknown")
        print(f"[CACHE] ✓  Local cache file read — {n} predictions (generated {generated_at})")
        return data
    except Exception as e:
        print(f"[CACHE] ✗  Could not read local cache file: {e}")
        return None


def save_live_cache(predictions: list, forecast_dates: list[str]):
    """
    Saves live-inference predictions to the local file cache.
    Called by prediction_service after _run_live_inference completes,
    so subsequent requests with the same forecast dates skip inference entirely.
    """
    artifact = {
        "generated_at": datetime.utcnow().isoformat(),
        "source": "live_inference",
        "forecast_dates": forecast_dates,
        "predictions": predictions,
    }
    try:
        _LOCAL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_LOCAL_CACHE_PATH, "w") as f:
            json.dump(artifact, f, indent=2)

        # Also update in-memory so the next request within this session hits memory
        global _cached_artifact
        _cached_artifact = artifact

        # print(f"Live inference results saved to local cache ({len(predictions)} predictions)")
        print(f"[CACHE] ✓  Live inference results saved to local cache — {len(predictions)} predictions")
        print(f"[CACHE]    Path: {_LOCAL_CACHE_PATH}")
    except Exception as e:
        print(f"[CACHE] ✗  Could not save local cache: {e}")


# ── Cache validation ─────────────────────────────────────────────────────────

def _extract_matching_predictions(artifact: dict, current_dates: list[str], source_label: str) -> Optional[list[dict]]:
    """
    Returns predictions from an artifact if it covers all current_dates.
    Returns None if the artifact is stale (missing dates).
    """
    cached_dates = set(artifact.get("forecast_dates", []))
    requested_dates = set(current_dates)

    if not requested_dates.issubset(cached_dates):
        missing = requested_dates - cached_dates
        # print(f"Cache stale: {len(missing)} date(s) not covered: {sorted(missing)}")
        print(f"[CACHE] ✗  {source_label} is stale — {len(missing)} date(s) missing: {sorted(missing)}")

        return None

    cached_preds = {p["date"]: p for p in artifact["predictions"]}
    result = [cached_preds[d] for d in current_dates if d in cached_preds]
    return result


# ── Main cache lookup ─────────────────────────────────────────────────────────

def get_cached_predictions(current_forecast_dates: list[str]) -> Optional[list[dict]]:
    """
    Returns cached predictions if they cover exactly the current forecast dates.
    Returns None if the cache is stale or unavailable — caller must run live inference.

    Cache invalidation logic:
    - Different model version activated → re-download artifact
    - New forecast dates appeared (daily data update) → cache is stale
    - Artifact missing from MLflow → fall back to live inference

    Returns cached predictions if available and fresh, otherwise None.

    Lookup order:
    1. In-memory (fastest — already fetched this session)
    2. MLflow artifact (version-specific, authoritative)
    3. Local file cache (written after live inference)
    4. None → caller must run live inference
    """
    global _cached_artifact, _cached_for_version

    active_version = get_active_version(MODEL_MULTIVARIATE)

    n_requested = len(current_forecast_dates)
    print(f"[CACHE] →  Lookup for {n_requested} date(s) | active model version: v{active_version}")

    # ── Tier 1: in-memory ─────────────────────────────────────────────────────
    if _cached_for_version == active_version and _cached_artifact:
        result = _extract_matching_predictions(_cached_artifact, current_forecast_dates, "in-memory cache")
        if result is not None:
            print(f"[CACHE] ✓  HIT  Tier 1 (in-memory) — serving {len(result)} predictions")
            return result
        print(f"[CACHE] ✗  MISS Tier 1 (in-memory is stale)")

    # ── Tier 2: MLflow artifact ───────────────────────────────────────────────
    if _cached_for_version != active_version:
        print(f"[CACHE] →  Version changed: {_cached_for_version} → v{active_version}. Fetching MLflow artifact...")
        run_id = _resolve_run_id(active_version)
        mlflow_artifact = _download_artifact(run_id) if run_id else None

        _cached_artifact = mlflow_artifact
        _cached_for_version = active_version

        if mlflow_artifact:
            result = _extract_matching_predictions(mlflow_artifact, current_forecast_dates,
                                                   f"MLflow artifact (v{active_version})")
            if result is not None:
                print(f"[CACHE] ✓  HIT  Tier 2 (MLflow v{active_version}) — serving {len(result)} predictions")
                return result
            print(f"[CACHE] ✗  MISS Tier 2 (MLflow artifact is stale)")
    else:
        print(f"[CACHE] ✗  MISS Tier 2 (MLflow artifact not available)")

    # ── Tier 3: local file cache ──────────────────────────────────────────────
    print(f"[CACHE] →  Checking Tier 3 (local file cache)...")
    local = _read_local_cache()
    if local:
        result = _extract_matching_predictions(local, current_forecast_dates, "local file cache")
        if result is not None:
            print(f"[CACHE] ✓  HIT  Tier 3 (local file) — serving {len(result)} predictions")
            _cached_artifact = local  # promote to in-memory
            return result
        print(f"[CACHE] ✗  MISS Tier 3 (local file is stale)")

    print(f"[CACHE] ✗  MISS All tiers — live inference required")
    return None


def invalidate_cache():
    """Clears all cache tiers. Called after model activation."""
    global _cached_artifact, _cached_for_version

    print(f"[CACHE] ⚡  Invalidating all cache tiers...")

    _cached_artifact = None
    _cached_for_version = None

    print(f"[CACHE]    Tier 1 (in-memory) cleared")

    if _LOCAL_CACHE_PATH.exists():
        try:
            _LOCAL_CACHE_PATH.unlink()
            print(f"[CACHE]    Tier 3 (local file) deleted: {_LOCAL_CACHE_PATH}")
        except Exception as e:
            print(f"[CACHE] ✗  Could not delete local cache file: {e}")
    else:
        print(f"[CACHE]    Tier 3 (local file) — nothing to delete")

    print(f"[CACHE] ✓  Cache fully invalidated")
