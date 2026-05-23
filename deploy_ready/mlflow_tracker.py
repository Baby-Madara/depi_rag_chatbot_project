"""
CityFoam — MLflow Tracker
===========================
Central integration point between CityFoam's runtime and MLflow.

Two modes of operation
-----------------------
1. PRODUCTION (server is running, MLFLOW_TRACKING_URI is set)
   Every chat query is logged as an MLflow run under the
   "cityfoam-production" experiment:
     • params:  language, optimized_query (first 120 chars)
     • metrics: rag_score, latency_ms, response_chars, rag_poor (0/1)
     • tags:    chat_id, env

2. EXPERIMENT (run_rag_experiment.py is invoked directly)
   A parent run wraps many child runs, one per hyperparameter combination.
   Each child logs ingest params + retrieval metrics so you can compare
   configurations side-by-side in the MLflow UI / Azure ML Studio.

Azure ML connection
-------------------
Set MLFLOW_TRACKING_URI to your Azure ML workspace URI:

    azureml://eastus.api.azureml.ms/mlflow/v1.0/subscriptions/<SUB>/
    resourceGroups/<RG>/providers/Microsoft.MachineLearningServices/
    workspaces/<WS>

When running inside Azure ML (AML_RUN_ID is set), the URI is picked up
automatically — no manual configuration needed.

Local fallback
--------------
Leave MLFLOW_TRACKING_URI unset → runs are stored in ./mlruns/ on disk.
"""

import logging
import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Dict, Optional

logger = logging.getLogger("cityfoam.mlflow")

# ---------------------------------------------------------------------------
# Lazy MLflow import — so the server starts even if mlflow is not installed
# ---------------------------------------------------------------------------
_mlflow_available: Optional[bool] = None


def _mlflow():
    """Return the mlflow module, or None if not installed."""
    global _mlflow_available
    if _mlflow_available is False:
        return None
    try:
        # pyrefly: ignore [missing-import]
        import mlflow  # noqa: PLC0415
        _mlflow_available = True
        return mlflow
    except ImportError:
        _mlflow_available = False
        logger.warning("mlflow not installed — tracking disabled. Run: pip install mlflow")
        return None


# ---------------------------------------------------------------------------
# Experiment names
# ---------------------------------------------------------------------------
PROD_EXPERIMENT   = "cityfoam-production"
INGEST_EXPERIMENT = "cityfoam-rag-experiments"
EVAL_EXPERIMENT   = "cityfoam-rag-evaluation"


# ---------------------------------------------------------------------------
# Initialisation — called once at server startup from server.py lifespan
# ---------------------------------------------------------------------------
def init_mlflow(env: str = "production") -> None:
    """
    Configure the MLflow tracking URI and set the active experiment.

    Priority for tracking URI:
      1. MLFLOW_TRACKING_URI env var (Azure ML URI or remote server)
      2. ./mlruns  (local default)
    """
    ml = _mlflow()
    if not ml:
        return

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if tracking_uri:
        ml.set_tracking_uri(tracking_uri)
        logger.info("MLflow tracking URI: %s", tracking_uri[:80])
    else:
        logger.info("MLflow tracking URI: local (./mlruns)")

    experiment_name = PROD_EXPERIMENT if env == "production" else INGEST_EXPERIMENT
    ml.set_experiment(experiment_name)
    logger.info("MLflow active experiment: %s", experiment_name)


# ---------------------------------------------------------------------------
# Per-query logging — called from server.py after each streaming response
# ---------------------------------------------------------------------------
def log_query_to_mlflow(
    *,
    chat_id:         str,
    raw_query:       str,
    optimized_query: str,
    language:        str,
    rag_score:       float,
    rag_poor:        bool,
    latency_ms:      float,
    response_chars:  int,
    n_chunks:        int   = 5,
    extra_tags:      Optional[Dict[str, str]] = None,
) -> None:
    """
    Log a single query turn as a short MLflow run.
    This is intentionally lightweight — it runs in a BackgroundTask.
    """
    ml = _mlflow()
    if not ml:
        return

    try:
        with ml.start_run(
            run_name=f"query-{chat_id[:8]}",
            tags={
                "chat_id": chat_id,
                "env":     os.environ.get("ENV", "production"),
                **(extra_tags or {}),
            },
        ):
            # Parameters — things you chose / configured
            ml.log_params({
                "language":        language,
                "n_chunks":        n_chunks,
                "query_preview":   (optimized_query or raw_query)[:120],
            })

            # Metrics — numbers you want to track over time
            ml.log_metrics({
                "rag_score":      round(rag_score, 4),
                "rag_poor":       int(rag_poor),
                "latency_ms":     round(latency_ms, 1),
                "response_chars": response_chars,
            })

    except Exception as exc:
        # Never let MLflow errors break the main request flow
        logger.error("MLflow log_query failed: %s", exc)


# ---------------------------------------------------------------------------
# RAG evaluation batch logging — called from run_rag_experiment.py
# ---------------------------------------------------------------------------
@contextmanager
def experiment_run(
    run_name: str,
    params:   Dict[str, Any],
    experiment_name: str = INGEST_EXPERIMENT,
    nested: bool = False,
):
    """
    Context manager for a single experiment run.
    Logs params on entry, flushes metrics on exit.

    Usage:
        with experiment_run("chunk800-k5", params={"chunk_size": 800}) as run:
            run.log_metric("mean_rag_score", 0.72)
    """
    ml = _mlflow()
    if not ml:
        # Yield a no-op stub so callers don't need to check
        yield _NullRun()
        return

    ml.set_experiment(experiment_name)
    with ml.start_run(run_name=run_name, nested=nested) as active_run:
        ml.log_params(params)
        yield _ActiveRun(ml, active_run)


class _ActiveRun:
    """Thin wrapper so callers can log metrics without importing mlflow."""
    def __init__(self, ml, run):
        self._ml  = ml
        self._run = run

    def log_metric(self, key: str, value: float, step: int = None) -> None:
        self._ml.log_metric(key, value, step=step)

    def log_metrics(self, d: Dict[str, float], step: int = None) -> None:
        self._ml.log_metrics(d, step=step)

    def log_artifact(self, path: str) -> None:
        self._ml.log_artifact(path)

    def set_tag(self, key: str, value: str) -> None:
        self._ml.set_tag(key, value)

    @property
    def run_id(self) -> str:
        return self._run.info.run_id


class _NullRun:
    """No-op stub used when MLflow is unavailable."""
    run_id = "no-mlflow"

    def log_metric(self, *a, **kw):   pass
    def log_metrics(self, *a, **kw):  pass
    def log_artifact(self, *a, **kw): pass
    def set_tag(self, *a, **kw):      pass


# ---------------------------------------------------------------------------
# Model registry helpers — called from model_registry.py
# ---------------------------------------------------------------------------
def register_model(model_uri: str, name: str, tags: Optional[Dict] = None) -> str:
    """
    Register a logged model in the MLflow Model Registry.
    Returns the model version string.
    """
    ml = _mlflow()
    if not ml:
        logger.warning("MLflow unavailable — model not registered.")
        return "unregistered"

    result = ml.register_model(model_uri=model_uri, name=name, tags=tags)
    logger.info("Registered model '%s' version %s", name, result.version)
    return result.version


def get_latest_model_version(name: str, stage: str = "Production") -> Optional[str]:
    """Return the latest model version in the given stage, or None."""
    ml = _mlflow()
    if not ml:
        return None
    try:
        client   = ml.tracking.MlflowClient()
        versions = client.get_latest_versions(name, stages=[stage])
        return versions[0].version if versions else None
    except Exception as exc:
        logger.error("Could not fetch model version: %s", exc)
        return None
