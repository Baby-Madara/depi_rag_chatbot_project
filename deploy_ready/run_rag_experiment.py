"""
CityFoam — RAG Hyperparameter Experiment Runner (v2)
=====================================================
Sweeps chunk_size, chunk_overlap, n_results and logs every combination
as a nested MLflow run under one parent so you can compare in MLflow UI
or Azure ML Studio.

Scoring uses rag_eval.score_distances() — the calibrated relative-gap
scorer, NOT the old exponential formula.  distance_scale is removed from
the grid because it is internal to rag_eval and already calibrated.

Run:
    docker exec -it cityfoam-app python run_rag_experiment.py

View results:
    docker compose --profile mlflow up   →  http://localhost:5000
"""

import itertools
import logging
import os
import tempfile
import time

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

import ingest as _ingest_mod
from ingest import normalize_arabic, process_data_folder
from mlflow_tracker import INGEST_EXPERIMENT, experiment_run
from rag_eval import WARNING_THRESHOLD, score_distances

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cityfoam.experiment")

# ---------------------------------------------------------------------------
# Hyperparameter grid  (distance_scale removed — handled in rag_eval.py)
# ---------------------------------------------------------------------------
PARAM_GRID = {
    "chunk_size":    [600, 900, 1200],
    "chunk_overlap": [100, 200],
    "n_results":     [3, 5, 7],
}

# ---------------------------------------------------------------------------
# Eval queries — must reflect actual customer language and KB content
# ---------------------------------------------------------------------------
EVAL_QUERIES = [
    # Arabic
    {"query": "ما هو سعر مرتبة بيرلا بوكيت",      "lang": "ar"},
    {"query": "كم يستغرق الشحن للقاهرة",            "lang": "ar"},
    {"query": "هل يمكنني ارجاع المرتبه بعد 45 يوم", "lang": "ar"},
    {"query": "ما هو الضمان على المراتب",            "lang": "ar"},
    {"query": "فروع سيتي فوم",                       "lang": "ar"},
    {"query": "سعر مرتبه اريجاتو",                   "lang": "ar"},
    # English
    {"query": "120 night sleep trial policy",         "lang": "en"},
    {"query": "return pillow after opening",          "lang": "en"},
    {"query": "warranty claim process",               "lang": "en"},
    {"query": "white glove delivery Banha",           "lang": "en"},
    {"query": "Signature Hybrid mattress trial",      "lang": "en"},
    {"query": "branch locations Cairo Alexandria",    "lang": "en"},
]

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


# ---------------------------------------------------------------------------
# Build a temp ChromaDB collection for one chunk configuration
# ---------------------------------------------------------------------------
def _build_temp_collection(chunk_size: int, chunk_overlap: int,
                           tmp_dir: str, model: SentenceTransformer):
    logger.info("  chunk=%d  overlap=%d", chunk_size, chunk_overlap)

    # Temporarily override the module-level splitter in ingest.py
    original = _ingest_mod.TEXT_SPLITTER
    _ingest_mod.TEXT_SPLITTER = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs = process_data_folder()
    _ingest_mod.TEXT_SPLITTER = original   # always restore

    client     = chromadb.PersistentClient(path=tmp_dir)
    collection = client.create_collection(name="exp_collection")
    texts      = [d["text"] for d in docs]
    embeddings = model.encode(texts, show_progress_bar=False).tolist()
    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=[d["metadata"] for d in docs],
        ids=[f"doc_{i}" for i in range(len(docs))],
    )
    return collection, len(docs)


# ---------------------------------------------------------------------------
# Score one configuration
# ---------------------------------------------------------------------------
def _evaluate(collection: chromadb.Collection,
              model: SentenceTransformer,
              n_results: int) -> dict:
    gap_scores  = []
    top1_dists  = []
    poor_count  = 0

    for item in EVAL_QUERIES:
        vector    = model.encode([normalize_arabic(item["query"])]).tolist()
        result    = collection.query(
            query_embeddings=vector,
            n_results=n_results,
            include=["distances"],
        )
        distances = result.get("distances", [[]])[0]

        if not distances:
            gap_scores.append(0.0)
            poor_count += 1
            continue

        gap_score, _ = score_distances(distances)
        gap_scores.append(gap_score)
        top1_dists.append(distances[0])
        if gap_score < WARNING_THRESHOLD:
            poor_count += 1

    n = len(EVAL_QUERIES)
    return {
        "mean_rag_score": round(sum(gap_scores) / n, 4),
        "pct_poor":       round(100 * poor_count / n, 1),
        "mean_top1_dist": round(sum(top1_dists) / max(len(top1_dists), 1), 4),
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def run_experiment():
    import mlflow

    combos = list(itertools.product(*PARAM_GRID.values()))
    keys   = list(PARAM_GRID.keys())
    logger.info("Sweep: %d combinations × %d queries each.", len(combos), len(EVAL_QUERIES))

    # Load embedding model once — reused across all runs
    logger.info("Loading embedding model...")
    shared_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    mlflow.set_experiment(INGEST_EXPERIMENT)

    with mlflow.start_run(run_name="rag-sweep") as parent:
        mlflow.set_tag("n_eval_queries",  len(EVAL_QUERIES))
        mlflow.set_tag("embedding_model", EMBEDDING_MODEL_NAME)
        mlflow.set_tag("scorer",          "relative_gap")

        best_score  = -1.0
        best_params = {}
        best_run_id = None

        for i, combo in enumerate(combos, 1):
            params   = dict(zip(keys, combo))
            run_name = (f"c{params['chunk_size']}"
                        f"-o{params['chunk_overlap']}"
                        f"-k{params['n_results']}")
            logger.info("[%d/%d]  %s", i, len(combos), run_name)

            with tempfile.TemporaryDirectory() as tmp_dir:
                t0 = time.perf_counter()
                collection, n_chunks = _build_temp_collection(
                    params["chunk_size"], params["chunk_overlap"],
                    tmp_dir, shared_model,
                )
                ingest_secs = round(time.perf_counter() - t0, 1)

                metrics = _evaluate(collection, shared_model, params["n_results"])
                metrics["ingest_secs"] = ingest_secs
                metrics["n_chunks"]    = n_chunks

            with experiment_run(run_name, params, INGEST_EXPERIMENT, nested=True) as run:
                run.log_metrics(metrics)
                run.set_tag("parent_run_id", parent.info.run_id)

            logger.info("  score=%.4f  poor=%.1f%%  chunks=%d",
                        metrics["mean_rag_score"], metrics["pct_poor"], n_chunks)

            if metrics["mean_rag_score"] > best_score:
                best_score  = metrics["mean_rag_score"]
                best_run_id = run.run_id
                best_params = {**params, **metrics}

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.set_tag("best_child_run_id", best_run_id or "none")
        logger.info("Best config  score=%.4f: %s", best_score, best_params)

    logger.info("Done. View: docker compose --profile mlflow up → http://localhost:5000")


if __name__ == "__main__":
    run_experiment()