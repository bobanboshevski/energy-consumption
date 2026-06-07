import os
import mlflow

from app.core.config import settings


def setup_mlflow():
    os.environ["MLFLOW_TRACKING_USERNAME"] = settings.MLFLOW_TRACKING_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = settings.MLFLOW_TRACKING_PASSWORD
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)


def get_registered_models():
    setup_mlflow()
    client = mlflow.tracking.MlflowClient()
    models = client.search_registered_models()
    result = []
    for m in models:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            all_versions = client.search_model_versions(f"name='{m.name}'")
        result.append({
            "name": m.name,
            "versions": [
                {
                    "version": v.version,
                    "stage": v.current_stage,
                    "run_id": v.run_id,
                    "status": v.status,
                }
                for v in sorted(all_versions, key=lambda x: int(x.version), reverse=True)
            ]
        })
    return result


def get_experiment_runs(experiment_name: str):
    setup_mlflow()
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if not exp:
        return []

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=20
    )

    # I set order, so the UI will be more readable
    METRIC_ORDER = [
        "test_mae", "test_rmse", "test_mse",
        "full_mae", "full_rmse", "full_mse",
        "val_loss", "loss", "validation_loss",
        "stopped_epoch", "restored_epoch",
    ]
    PARAM_ORDER = [
        "target_col", "window_size", "test_size", "feature_cols",
        "epochs", "batch_size", "validation_split",
        "monitor", "patience", "restore_best_weights",
    ]

    return [
        {
            "run_id": r.info.run_id,
            "run_name": r.info.run_name,
            "status": r.info.status,
            "start_time": r.info.start_time,
            "metrics": sort_dict(r.data.metrics, METRIC_ORDER),
            "params": sort_dict(r.data.params, PARAM_ORDER),
        }
        for r in runs
    ]


def transition_model_version(model_name: str, version: str, stage: str):
    """Transitions a model version to a new stage (Staging/Production/Archived)."""
    setup_mlflow()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
    return {"success": True, "model": model_name, "version": version, "stage": stage}


def sort_dict(d: dict, preferred_order: list) -> dict:
    ordered_keys = [k for k in preferred_order if k in d]
    remaining_keys = sorted(k for k in d if k not in preferred_order)
    return {k: d[k] for k in ordered_keys + remaining_keys}


def get_all_model_versions(model_name: str):
    """Returns ALL versions of a model enriched with run metrics."""
    setup_mlflow()
    client = mlflow.tracking.MlflowClient()

    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except Exception as e:
        return {"error": str(e)}

    METRIC_ORDER = ["test_mae", "test_rmse", "full_mae", "full_rmse"]
    PARAM_ORDER = ["target_col", "window_size", "test_size", "feature_cols"]

    result = []
    for v in sorted(versions, key=lambda x: int(x.version), reverse=True):
        metrics = {}
        params = {}
        try:
            run = client.get_run(v.run_id)
            # Only include the key metrics we care about
            key_metrics = ["test_mae", "test_rmse", "full_mae", "full_rmse"]
            metrics = {k: round(v2, 4) for k, v2 in run.data.metrics.items() if k in key_metrics}
            key_params = ["window_size", "test_size", "target_col", "feature_cols"]
            params = {k: v2 for k, v2 in run.data.params.items() if k in key_params}
        except Exception:
            pass

        result.append({
            "version": v.version,
            "stage": v.current_stage,
            "run_id": v.run_id,
            "status": v.status,
            "creation_timestamp": v.creation_timestamp,
            "metrics": sort_dict(metrics, METRIC_ORDER),
            "params": sort_dict(params, PARAM_ORDER),
        })

    return result


# todo: i can reuse this in many parts of the code
def get_run_id_for_version(model_name: str, version: str) -> str | None:
    """
    Resolves a model version number to its MLflow run_id.

    This is the bridge between the human-readable version number (e.g. "32")
    and the UUID run_id under which all artifacts are physically stored in MLflow.

    Returns:
        run_id string if found, None if the version does not exist.
    """
    try:
        setup_mlflow()
        client = mlflow.tracking.MlflowClient()
        mv = client.get_model_version(model_name, version)
        return mv.run_id
    except Exception:
        return None


def get_latest_version(model_name: str) -> str | None:
    """
    Returns the highest (newest) version number for a registered model as a string,
    or None if the model has no versions or MLflow is unreachable.

    Single source of truth for resolving the "latest" alias to a concrete
    MLflow version number — used by both model loading and explainability.
    """
    try:
        setup_mlflow()
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            return None
        return max(versions, key=lambda v: int(v.version)).version
    except Exception as e:
        print(f"WARNING: Could not resolve latest version for {model_name}: {e}")
        return None
