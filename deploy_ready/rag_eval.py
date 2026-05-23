"""
CityFoam — RAG Quality Evaluation
====================================
After each ChromaDB query we score how relevant the retrieved chunks are.

Scoring method
--------------
ChromaDB returns L2 distances by default. For the paraphrase-multilingual-
MiniLM-L12-v2 model in 384-dimensional space, typical L2 distances between
semantically related chunks fall in the range 7–10, and unrelated chunks
land at 12–20+.

We use two complementary approaches:

1. Relative gap score (primary)
   Compares the best match distance against the worst match distance.
   A large gap means the top result is meaningfully closer than the rest.
   score = 1 - (best_dist / worst_dist)
   Range: 0 (no gap, all equally bad) → 1 (perfect separation)

2. Exponential decay (secondary, for logging)
   score = exp(-mean_top3_dist / DISTANCE_SCALE)
   DISTANCE_SCALE = 10.0  calibrated for L2 distances in this embedding space.

The WARNING_THRESHOLD (default 0.25) flags queries where retrieved chunks
are unlikely to contain a useful answer.
"""

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger("cityfoam.rag_eval")

# Calibrated for paraphrase-multilingual-MiniLM-L12-v2 L2 distances
# Typical good retrieval: distances ~7-9  → decay_score ~0.40-0.50
# Typical poor retrieval: distances ~14+  → decay_score ~0.25 or below
DISTANCE_SCALE    = 10.0   # was 1.0 — recalibrated for 384-dim L2 space
WARNING_THRESHOLD = 0.25   # flag if relative gap score is below this
TOP_K_FOR_SCORING = 3


@dataclass
class RetrievalResult:
    documents:     list[str]
    distances:     list[float]
    quality_score: float = 1.0
    decay_score:   float = 1.0   # exp decay score, for MLflow logging
    is_poor:       bool  = False
    context_str:   str   = ""


def score_distances(distances: list[float]) -> tuple[float, float]:
    """
    Returns (relative_gap_score, decay_score) both in [0, 1].

    relative_gap_score:
        How much better is the best match vs the worst?
        1.0 = best match is perfectly closer; 0.0 = no discrimination.

    decay_score:
        exp(-mean_top3 / DISTANCE_SCALE), calibrated for L2 in 384-dim space.
    """
    if not distances:
        return 0.0, 0.0

    sorted_d   = sorted(distances)
    best_dist  = sorted_d[0]
    worst_dist = sorted_d[-1]
    top_k      = sorted_d[:TOP_K_FOR_SCORING]
    mean_top_k = sum(top_k) / len(top_k)

    # Relative gap — main quality signal
    if worst_dist == 0:
        gap_score = 1.0
    else:
        gap_score = 1.0 - (best_dist / worst_dist)

    # Exponential decay — secondary signal, useful for trend monitoring
    decay_score = math.exp(-mean_top_k / DISTANCE_SCALE)

    return round(gap_score, 4), round(decay_score, 4)


def evaluate_retrieval(
    documents: list[str],
    distances: list[float],
    query: str = "",
) -> RetrievalResult:
    """
    Score retrieval quality and return a RetrievalResult.
    quality_score is the relative gap score (primary).
    decay_score is the exponential score (logged to MLflow for trends).
    """
    gap_score, decay_score = score_distances(distances)
    is_poor = gap_score < WARNING_THRESHOLD

    if is_poor:
        logger.warning(
            "Poor RAG retrieval — gap=%.3f  decay=%.3f  query=%r  "
            "top distances=%s",
            gap_score, decay_score, (query or "")[:80],
            [round(d, 2) for d in sorted(distances)[:TOP_K_FOR_SCORING]],
        )
    else:
        logger.debug(
            "RAG quality gap=%.3f decay=%.3f query=%r",
            gap_score, decay_score, (query or "")[:60],
        )

    context_str = "\n\n---\n\n".join(documents) if documents else ""

    if is_poor and context_str:
        context_str += (
            "\n\n[SYSTEM NOTE: Retrieval confidence is LOW. "
            "If the context above does not clearly answer the question, "
            "say so and offer to connect the customer with a human agent.]"
        )

    return RetrievalResult(
        documents=documents,
        distances=distances,
        quality_score=gap_score,
        decay_score=decay_score,
        is_poor=is_poor,
        context_str=context_str,
    )