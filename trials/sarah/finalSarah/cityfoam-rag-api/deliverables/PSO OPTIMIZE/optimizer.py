# =============================================================================
# CityFoam RAG API — PSO Optimizer Module
# Production-Grade | PySwarms | APScheduler | Atomic Writes | Dynamic Data
# =============================================================================
import os
import sys
import json
import logging
import subprocess
import numpy as np
import chromadb
from typing import Dict, List
from datetime import datetime
from sentence_transformers import SentenceTransformer
from pyswarms.single.global_best import GlobalBestPSO
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
# =============================================================================
# SECTION 1 — LOGGING
# Outputs to BOTH terminal (stdout) and a rotating local log file.
# Every line is timestamped. Import this logger in every function below.
# =============================================================================

LOG_FILE = "pso_optimizer.log"

# Build a single logger that fans out to two handlers simultaneously
_formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

_file_handler   = logging.FileHandler(LOG_FILE, encoding="utf-8")
_file_handler.setFormatter(_formatter)

_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setFormatter(_formatter)

logger = logging.getLogger("pso_optimizer")
logger.setLevel(logging.INFO)
logger.addHandler(_file_handler)
logger.addHandler(_stream_handler)
# Prevent duplicate lines if root logger is also configured
logger.propagate = False


# =============================================================================
# SECTION 2 — PSO CONFIGURATION
# Plain constants consumed directly by GlobalBestPSO — no custom dataclasses.
# =============================================================================

PSO_OPTIONS = {
    "c1": 1.5,   # cognitive parameter — personal best pull
    "c2": 1.5,   # social parameter    — global best pull
    "w":  0.7,   # inertia weight
}

# Bounds: (min_array, max_array) — PySwarms tuple format
# Search space  → [top_k, similarity_threshold, chunk_weight]
SEARCH_BOUNDS = (
    np.array([1,   0.0, 0.0]),
    np.array([20,  1.0, 1.0]),
)

# Ranking space → [semantic_weight, keyword_weight, recency_weight, authority_weight]
RANKING_BOUNDS = (
    np.array([0.0, 0.0, 0.0, 0.0]),
    np.array([1.0, 1.0, 1.0, 1.0]),
)

NUM_PARTICLES  = 10
NUM_ITERATIONS = 20

# File paths
USER_QUERIES_FILE = "./user_queries.jsonl"
PSO_PARAMS_FILE   = "./pso_best_params.json"
PSO_TMP_FILE      = "./pso_best_params.tmp.json"


# =============================================================================
# SECTION 3 — DYNAMIC DATA LOADER
# Reads real user interaction logs from user_queries.jsonl.
# Falls back to hardcoded CityFoam mock data on any failure — never crashes.
# =============================================================================

FALLBACK_DATA = [
    {"query": "ما هو سعر منتج بيرلا بوكيت؟",    "expected": "بيرلا بوكيت"},
    {"query": "أين تقع فروع CityFoam؟",           "expected": "CityFoam"},
    {"query": "كيف أسترجع منتجاً معيباً؟",        "expected": "استرجاع"},
    {"query": "هل هناك خصومات متاحة على أريجاتو؟","expected": "خصم"},
    {"query": "بكام مخده الرقبه",                 "expected": "الرقبه"},
    {"query": "do you have contour memory pillows?",   "expected": "contour memory pillows"},
    {"query": "What is the return policy?",        "expected": "return"},
    {"query": "Where can I find product details?", "expected": "product"},
]


def load_real_user_data() -> tuple:
    """
    Dynamically loads query/expected pairs from user_queries.jsonl
    located in the project root directory.
    
    Returns:
        tuple: (queries: List[str], expected_answers: List[str])
    """
    try:
        # Get the absolute directory of the current optimizer.py file
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        # If optimizer.py is inside a 'src' subdirectory, step up to the project root
        if os.path.basename(BASE_DIR) == "src":
            PROJECT_ROOT = os.path.dirname(BASE_DIR)
        else:
            PROJECT_ROOT = BASE_DIR
            
        # Construct the absolute path to the target JSONL file
        TARGET_FILE = os.path.join(PROJECT_ROOT, "user_queries.jsonl")
        
        logger.info(f"[DataLoader] Attempting to load from absolute path: {TARGET_FILE}")

        queries = []
        expected = []
        
        # Read the JSONL file line by line
        with open(TARGET_FILE, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        item = json.loads(line.strip())
                        if "query" in item and "expected" in item:
                            queries.append(item["query"])
                            expected.append(item["expected"])
                    except json.JSONDecodeError as je:
                        logger.error(f"[DataLoader] Syntax error in JSONL at line {line_num}: {je}")

        # If no queries were successfully parsed, raise an error to trigger fallback
        if not queries:
            raise ValueError(f"No valid query/expected pairs found in '{TARGET_FILE}'.")

        logger.info(
            f"[DataLoader] SUCCESS: Loaded {len(queries)} samples from '{TARGET_FILE}'"
        )
        return queries, expected

    except Exception as e:
        logger.warning(
            f"[DataLoader] Failed to load custom file: {e}. "
            f"Falling back to {len(FALLBACK_DATA)} mock samples."
        )
        queries  = [item["query"]    for item in FALLBACK_DATA]
        expected = [item["expected"] for item in FALLBACK_DATA]
        return queries, expected

# =============================================================================
# SECTION 4 — FITNESS EVALUATOR
# Internal scoring logic is PRESERVED EXACTLY from the previous version.
# Only the constructor and ChromaDB wiring are updated to use Config centrally.
# =============================================================================

class RAGFitnessEvaluator:
    """
    PySwarms-compatible fitness evaluator backed by real ChromaDB retrieval.

    PySwarms contract:
        fitness_fn(position_matrix: np.ndarray[n_particles, n_dims])
            -> np.ndarray[n_particles]   (cost per particle, lower = better)

    The two public methods (evaluate_search_quality, evaluate_ranking_quality)
    satisfy this contract. Internal scoring (_score_*) is untouched.
    """

    def __init__(self, test_queries: List[str], expected_answers: List[str]):
        self.test_queries     = test_queries
        self.expected_answers = expected_answers

        logger.info(
            f"[Evaluator] Initializing with {len(test_queries)} samples..."
        )
        self.embedding_model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2"
        )

        # Central config wiring — Config.CHROMA_DB_DIR / Config.COLLECTION_NAME
        from config import Config
        chroma_client   = chromadb.PersistentClient(path=Config.CHROMA_DB_DIR)
        self.collection = chroma_client.get_collection(name=Config.COLLECTION_NAME)
        logger.info("[Evaluator] ChromaDB collection connected successfully.")

    # ------------------------------------------------------------------
    # INTERNAL SCORING — LOGIC PRESERVED EXACTLY, DO NOT MODIFY
    # ------------------------------------------------------------------

    def _score_search_particle(self, params: np.ndarray) -> float:
        """Score one particle for search parameters. Returns loss [0, 1]."""
        top_k         = int(np.clip(params[0], 1, 20))
        sim_threshold = float(np.clip(params[1], 0.0, 1.0))
        total_score   = 0.0

        for query, expected in zip(self.test_queries, self.expected_answers):
            query_vector = self.embedding_model.encode([query]).tolist()
            results = self.collection.query(
                query_embeddings=query_vector,
                n_results=top_k,
                include=["documents", "distances"]
            )
            for doc, dist in zip(results["documents"][0], results["distances"][0]):
                semantic_score = 1 / (1 + dist)
                if semantic_score >= sim_threshold:
                    if expected.lower() in doc.lower():
                        total_score += 1.0
                        break  # found — move to next query

        accuracy = total_score / max(len(self.test_queries), 1)
        return 1.0 - accuracy

    def _score_ranking_particle(self, params: np.ndarray) -> float:
        """Score one particle for ranking weights. Returns loss [0, ∞)."""
        weight_sum = params.sum()
        if weight_sum == 0:
            return 1.0
        weights = params / weight_sum   # normalize to sum to 1
        w_semantic, w_keyword, w_recency, w_authority = weights

        # Penalize extreme distributions — PRESERVED EXACTLY
        penalty = 0.0
        if np.max(weights) > 0.8:
            penalty += 0.5
        if np.min(weights) < 0.05:
            penalty += 0.3

        total_score = 0.0

        for query, expected in zip(self.test_queries, self.expected_answers):
            query_vector = self.embedding_model.encode([query]).tolist()
            results = self.collection.query(
                query_embeddings=query_vector,
                n_results=10,
                include=["documents", "distances", "metadatas"]
            )
            candidates = []
            for doc, dist, meta in zip(
                results["documents"][0],
                results["distances"][0],
                results["metadatas"][0]
            ):
                semantic_score  = 1 / (1 + dist)
                query_tokens    = set(query.lower().split())
                doc_tokens      = set(doc.lower().split())
                keyword_score   = len(query_tokens & doc_tokens) / max(len(query_tokens), 1)
                recency_score   = float(meta.get("recency",   0.5))
                authority_score = float(meta.get("authority", 0.5))

                final_score = (
                    w_semantic  * semantic_score  +
                    w_keyword   * keyword_score   +
                    w_recency   * recency_score   +
                    w_authority * authority_score
                )
                candidates.append((final_score, doc))

            if not candidates:
                continue
            candidates.sort(key=lambda x: x[0], reverse=True)
            top_doc = candidates[0][1]
            if expected.lower() in top_doc.lower():
                total_score += 1.0

        accuracy = total_score / max(len(self.test_queries), 1)
        return max(0.0, (1.0 - accuracy) + penalty)

    # ------------------------------------------------------------------
    # PUBLIC BATCH WRAPPERS — PySwarms interface
    # ------------------------------------------------------------------

    def evaluate_search_quality(self, position_matrix: np.ndarray) -> np.ndarray:
        """
        PySwarms fitness function for search parameters.
        Input:  (n_particles, 3)  → [top_k, similarity_threshold, chunk_weight]
        Output: (n_particles,)    → cost per particle
        """
        return np.array([
            self._score_search_particle(position_matrix[i])
            for i in range(position_matrix.shape[0])
        ])

    def evaluate_ranking_quality(self, position_matrix: np.ndarray) -> np.ndarray:
        """
        PySwarms fitness function for ranking weights.
        Input:  (n_particles, 4)  → [semantic, keyword, recency, authority]
        Output: (n_particles,)    → cost per particle
        """
        return np.array([
            self._score_ranking_particle(position_matrix[i])
            for i in range(position_matrix.shape[0])
        ])


# =============================================================================
# SECTION 5 — OPTIMIZATION RUNNERS
# GlobalBestPSO structural configurations PRESERVED EXACTLY.
# =============================================================================

class SearchParameterOptimizer:
    """Optimizes: top_k, similarity_threshold, chunk_weight via GlobalBestPSO."""

    @staticmethod
    def run(test_queries: List[str], test_answers: List[str]) -> Dict:
        logger.info("[SearchOpt] Starting Search Parameter Optimization...")

        evaluator = RAGFitnessEvaluator(test_queries, test_answers)

        optimizer = GlobalBestPSO(
            n_particles=NUM_PARTICLES,
            dimensions=3,
            options=PSO_OPTIONS,
            bounds=SEARCH_BOUNDS,
        )

        best_cost, best_position = optimizer.optimize(
            evaluator.evaluate_search_quality,
            iters=NUM_ITERATIONS,
            verbose=True,
        )

        best_params = {
            "top_k":                int(round(best_position[0])),
            "similarity_threshold": round(float(best_position[1]), 4),
            "chunk_weight":         round(float(best_position[2]), 4),
        }

        info = _convergence_info(optimizer.cost_history)
        logger.info(
            f"[SearchOpt] Complete | cost={best_cost:.6f} | "
            f"improvement={info.get('improvement_%', 0):.2f}% | "
            f"params={best_params}"
        )

        return {
            "optimization_type": "search_parameters",
            "best_params":        best_params,
            "best_cost":          float(best_cost),
            "cost_history":       optimizer.cost_history,
            "convergence_info":   info,
        }


class RankingWeightOptimizer:
    """Optimizes: semantic, keyword, recency, authority weights via GlobalBestPSO."""

    @staticmethod
    def run(test_queries: List[str], test_answers: List[str]) -> Dict:
        logger.info("[RankOpt] Starting Ranking Weight Optimization...")

        evaluator = RAGFitnessEvaluator(test_queries, test_answers)

        optimizer = GlobalBestPSO(
            n_particles=NUM_PARTICLES,
            dimensions=4,
            options=PSO_OPTIONS,
            bounds=RANKING_BOUNDS,
        )

        best_cost, best_position = optimizer.optimize(
            evaluator.evaluate_ranking_quality,
            iters=NUM_ITERATIONS,
            verbose=True,
        )

        # Weight normalization — PRESERVED EXACTLY
        weight_sum    = best_position.sum()
        best_position = best_position / weight_sum if weight_sum > 0 else best_position

        best_params = {
            "semantic_weight":  round(float(best_position[0]), 4),
            "keyword_weight":   round(float(best_position[1]), 4),
            "recency_weight":   round(float(best_position[2]), 4),
            "authority_weight": round(float(best_position[3]), 4),
        }

        info = _convergence_info(optimizer.cost_history)
        logger.info(
            f"[RankOpt] Complete | cost={best_cost:.6f} | "
            f"improvement={info.get('improvement_%', 0):.2f}% | "
            f"params={best_params}"
        )

        return {
            "optimization_type": "ranking_weights",
            "best_params":        best_params,
            "best_cost":          float(best_cost),
            "cost_history":       optimizer.cost_history,
            "convergence_info":   info,
        }


# =============================================================================
# SECTION 6 — UTILITIES
# =============================================================================

def _convergence_info(cost_history: list) -> Dict:
    """Summarizes convergence metrics from PySwarms cost_history."""
    if not cost_history:
        return {}
    return {
        "initial_cost":  round(cost_history[0],  6),
        "final_cost":    round(cost_history[-1],  6),
        "improvement_%": round(
            (cost_history[0] - cost_history[-1])
            / max(abs(cost_history[0]), 1e-9) * 100,
            2,
        ),
        "iterations": len(cost_history),
    }


def _to_serializable(obj):
    """JSON serializer for numpy floats and arrays."""
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_optimization_results(results: Dict, filename: str = None) -> str:
    """Save full combined results to a timestamped JSON file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"optimization_results_{timestamp}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=_to_serializable)

    logger.info(f"[Save] Full results -> {filename}")
    return filename


def load_optimization_results(filename: str) -> Dict:
    """Reload a previously saved optimization result."""
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_results(before: Dict, after: Dict) -> Dict:
    """Compare two optimization runs — before vs after."""
    comparison = {}
    if "best_params" in before and "best_params" in after:
        comparison["parameter_changes"] = {
            param: {
                "before": before["best_params"].get(param),
                "after":  after["best_params"].get(param),
            }
            for param in before["best_params"]
        }
    b_cost = before.get("best_cost", 0)
    a_cost = after.get("best_cost",  0)
    comparison["cost_improvement_%"] = round(
        (b_cost - a_cost) / max(abs(b_cost), 1e-9) * 100, 2
    )
    return comparison


def export_for_config(
    combined_results: Dict,
    target_file: str = PSO_PARAMS_FILE,
    tmp_file:    str = PSO_TMP_FILE,
) -> Dict:
    """
    ATOMIC WRITE — Zero-downtime safe file update.

    Flow:
      1. Flatten both optimizer outputs into the flat dict Config.load_pso_params() reads.
      2. Write the result to a TEMPORARY file (pso_best_params.tmp.json).
      3. Call os.replace(tmp_file, target_file) — a single atomic OS syscall
         that swaps the file in <1 ms with zero locking conflicts.

    This guarantees that a FastAPI worker reading pso_best_params.json mid-request
    will NEVER see a partially-written or corrupt file.
    """
    sp = combined_results["search_optimization"]["best_params"]
    rp = combined_results["ranking_optimization"]["best_params"]

    flat = {
        "top_k":                int(sp["top_k"]),
        "similarity_threshold": round(float(sp["similarity_threshold"]), 4),
        "chunk_weight":         round(float(sp["chunk_weight"]),         4),
        "weight_semantic":      round(float(rp["semantic_weight"]),      4),
        "weight_keyword":       round(float(rp["keyword_weight"]),       4),
        "weight_recency":       round(float(rp["recency_weight"]),       4),
        "weight_authority":     round(float(rp["authority_weight"]),     4),
    }

    # Step 1 — write to temp file
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(flat, f, indent=2, ensure_ascii=False)

    # Step 2 — atomic swap: tmp → production file (single OS syscall, no locking)
    os.replace(tmp_file, target_file)

    logger.info(f"[AtomicWrite] Config file updated atomically -> {target_file}")
    logger.info(f"[AtomicWrite] Final params: {json.dumps(flat)}")
    return flat


def plot_convergence(search_results: Dict, ranking_results: Dict):
    """Save convergence plots for both optimization phases."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(search_results["cost_history"])
        axes[0].set_title("Search Parameter Optimization — Cost History")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Cost (1 - accuracy)")
        axes[0].grid(True)

        axes[1].plot(ranking_results["cost_history"], color="orange")
        axes[1].set_title("Ranking Weight Optimization — Cost History")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Cost (1 - accuracy + penalty)")
        axes[1].grid(True)

        plt.tight_layout()
        plot_path = f"convergence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path)
        logger.info(f"[Plot] Convergence plot saved -> {plot_path}")
        plt.show()

    except ImportError:
        logger.warning("[Plot] matplotlib not installed — skipping convergence plot.")


# =============================================================================
# SECTION 7 — SCHEDULER ORCHESTRATOR & MAIN ENTRYPOINT
# Runs the optimizer on a background 1-hour interval using subprocess execution.
# =============================================================================

def trigger_pso_subprocess():
    """
    Spawns the optimizer as an independent OS background process.
    This offloads the heavy PySwarms CPU load from the main FastAPI threads.
    """
    logger.info("[Scheduler] Ticking... Spawning background PSO subprocess.")
    try:
        script_path = os.path.abspath(__file__)
        # Launching itself with a subprocess block
        subprocess.Popen([sys.executable, script_path, "--child-run"])
        logger.info("[Scheduler] PSO background subprocess detached successfully.")
    except Exception as e:
        logger.error(f"[Scheduler] Failed to trigger background process: {e}")

def start_hourly_scheduler():
    """
    Activates the interval scheduler. Call this ONCE during FastAPI startup.
    """
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        trigger_pso_subprocess,
        trigger=IntervalTrigger(hours=2),
        id="pso_hourly_job",
        replace_existing=True
    )
    scheduler.start()
    logger.info("[Scheduler] Continuous 2-Hour Swarm Scheduler is now ACTIVE.")
    return scheduler


if __name__ == "__main__":
    # Check if this run is triggered as a background child process by the scheduler
    if "--child-run" in sys.argv or len(sys.argv) == 1:
        logger.info("=" * 70)
        logger.info("CityFoam Swarm Optimization Subprocess Initiated (PySwarms)")
        logger.info("=" * 70)
        
        try:
            # 1. Load real logs or fall back
            queries, answers = load_real_user_data()
            
            # 2. Optimize Search Parameters
            search_res = SearchParameterOptimizer.run(queries, answers)
            
            # 3. Optimize Ranking Weights
            ranking_res = RankingWeightOptimizer.run(queries, answers)
            
            # 4. Save History and Atomically Deploy Configuration
            combined_results = {
                "timestamp": datetime.now().isoformat(),
                "search_optimization": search_res,
                "ranking_optimization": ranking_res,
            }
            save_optimization_results(combined_results)
            export_for_config(combined_results)
            
            # 5. Optional plotting
            plot_convergence(search_res, ranking_res)
            
            logger.info(" Optimization Cycle Finished Successfully. Subprocess exiting.")
            
        except Exception as e:
            logger.critical(f" Critical Failure during subprocess execution: {e}", exc_info=True)
            sys.exit(1)