"""
CityFoam — MLflow Model Registry
==================================
Manages the lifecycle of the SentenceTransformer embedding model used in the
RAG pipeline through the MLflow Model Registry.

Stages
------
  None       → model logged but not yet reviewed
  Staging    → candidate; passes eval but not in prod yet
  Production → live model used by the server
  Archived   → superseded; kept for rollback

Commands
--------
  python model_registry.py register   # log + register the current model
  python model_registry.py promote    # move Staging → Production
  python model_registry.py info       # print all registered versions
  python model_registry.py load       # smoke-test: load Production model

Azure ML notes
--------------
When MLFLOW_TRACKING_URI points to Azure ML, the registry is hosted in
Azure ML's model catalogue.  You can view / promote versions in:
  Azure ML Studio → Models → cityfoam-embedding
"""

import argparse
import logging
import os
import sys
import tempfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cityfoam.model_registry")

MODEL_NAME       = "cityfoam-embedding"
MODEL_ID         = "paraphrase-multilingual-MiniLM-L12-v2"
EXPERIMENT_NAME  = "cityfoam-model-registry"


# ---------------------------------------------------------------------------
# Register — log the model artifact + register in the MLflow registry
# ---------------------------------------------------------------------------
def register_embedding_model() -> str:
    """
    Log the SentenceTransformer as an MLflow artifact and register it.
    Returns the new model version string.
    """
    import mlflow
    import mlflow.pyfunc
    from sentence_transformers import SentenceTransformer

    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info("Loading model '%s' for registration ...", MODEL_ID)
    model = SentenceTransformer(MODEL_ID)

    # Wrap in a pyfunc so MLflow can load/serve it generically
    class _EmbeddingWrapper(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            from sentence_transformers import SentenceTransformer  # noqa
            self._model = SentenceTransformer.load(context.artifacts["model_path"])

        def predict(self, context, model_input):
            texts = model_input["text"].tolist()
            return self._model.encode(texts).tolist()

    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(tmp, "st_model")
        model.save(model_path)

        with mlflow.start_run(run_name=f"register-{MODEL_ID}") as run:
            mlflow.log_param("model_id",    MODEL_ID)
            mlflow.log_param("dimensions",  model.get_sentence_embedding_dimension())
            mlflow.log_param("max_seq_len", model.get_max_seq_length())

            mlflow.pyfunc.log_model(
                artifact_path="embedding_model",
                python_model=_EmbeddingWrapper(),
                artifacts={"model_path": model_path},
                registered_model_name=MODEL_NAME,
                pip_requirements=[f"sentence-transformers"],
            )
            version = _get_latest_version(MODEL_NAME)
            logger.info("Registered '%s' version %s (run=%s)", MODEL_NAME, version, run.info.run_id)
            return version


# ---------------------------------------------------------------------------
# Promote — move latest Staging version to Production
# ---------------------------------------------------------------------------
def promote_to_production() -> None:
    import mlflow
    client = mlflow.tracking.MlflowClient()

    staging_versions = client.get_latest_versions(MODEL_NAME, stages=["Staging"])
    if not staging_versions:
        logger.warning("No model version in Staging. Register one first.")
        return

    version = staging_versions[0].version

    # Archive any existing Production version first
    prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    for v in prod_versions:
        client.transition_model_version_stage(
            name=MODEL_NAME, version=v.version, stage="Archived"
        )
        logger.info("Archived previous Production version %s.", v.version)

    client.transition_model_version_stage(
        name=MODEL_NAME, version=version, stage="Production"
    )
    logger.info("Promoted version %s to Production.", version)


# ---------------------------------------------------------------------------
# Info — print all registered versions and their stages
# ---------------------------------------------------------------------------
def print_model_info() -> None:
    import mlflow
    client = mlflow.tracking.MlflowClient()

    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    except Exception:
        logger.error("Model '%s' not found in registry.", MODEL_NAME)
        return

    if not versions:
        logger.info("No versions registered for '%s'.", MODEL_NAME)
        return

    print(f"\n{'Version':<10} {'Stage':<14} {'Run ID':<36} {'Created'}")
    print("-" * 80)
    for v in sorted(versions, key=lambda x: int(x.version)):
        import datetime
        ts = datetime.datetime.fromtimestamp(v.creation_timestamp / 1000).strftime("%Y-%m-%d %H:%M")
        print(f"{v.version:<10} {v.current_stage:<14} {v.run_id:<36} {ts}")
    print()


# ---------------------------------------------------------------------------
# Load — smoke-test: load Production model and encode a test sentence
# ---------------------------------------------------------------------------
def load_and_test() -> None:
    import mlflow

    version = _get_latest_version(MODEL_NAME, stage="Production")
    if not version:
        logger.error("No Production version found. Promote one first.")
        return

    model_uri = f"models:/{MODEL_NAME}/Production"
    logger.info("Loading model from registry: %s", model_uri)
    model = mlflow.pyfunc.load_model(model_uri)

    import pandas as pd
    test_input = pd.DataFrame({"text": ["Hello, this is a test sentence."]})
    embeddings = model.predict(test_input)
    logger.info(
        "Embedding produced: dim=%d  first_5=%s",
        len(embeddings[0]),
        [round(x, 4) for x in embeddings[0][:5]],
    )
    logger.info("Model load test PASSED.")


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------
def _get_latest_version(name: str, stage: str = "None") -> str | None:
    import mlflow
    client   = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(name, stages=[stage])
    return versions[0].version if versions else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CityFoam MLflow Model Registry CLI")
    parser.add_argument(
        "command",
        choices=["register", "promote", "info", "load"],
        help="Action to perform",
    )
    args = parser.parse_args()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if tracking_uri:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)

    if args.command == "register":
        register_embedding_model()
    elif args.command == "promote":
        promote_to_production()
    elif args.command == "info":
        print_model_info()
    elif args.command == "load":
        load_and_test()
