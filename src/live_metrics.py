import os
import csv
import threading
import re
import uuid
from datetime import datetime
from typing import List, Optional, Tuple
import numpy as np

_write_lock = threading.Lock()

DEFAULT_LOG_FILENAME = "live_metrics_log.csv"

HEADER = [
    "interaction_id",
    "timestamp",
    "query",
    "response",
    "latency_ms",
    "error",
    "intent",
    "num_chunks_retrieved",
    "retrieval_quality",
    "hallucination_flag",
    "response_length",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "user_satisfaction"
]

def initialize_log_file(log_path: str) -> None:
    """إنشاء ملف CSV مع الرؤوس إذا لم يكن موجوداً."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if not os.path.isfile(log_path):
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)

def generate_interaction_id() -> str:
    """يُنتج معرّفاً فريداً للتفاعل."""
    return uuid.uuid4().hex[:12]

def compute_retrieval_quality(query: str, chunks: List[str], embed_model) -> float:
    """أعلى تشابه جيب تمام بين تضمين السؤال والمقاطع المسترجعة (0-100)."""
    if not chunks:
        return 0.0
    query_emb = embed_model.encode([query])[0]
    chunk_embs = embed_model.encode(chunks)
    dot = np.dot(chunk_embs, query_emb)
    norms = np.linalg.norm(chunk_embs, axis=1) * np.linalg.norm(query_emb) + 1e-8
    similarities = dot / norms
    return round(float(np.max(similarities)) * 100, 2)

def detect_hallucination(answer: str, chunks: List[str]) -> bool:
    """كشف هلوسة بسيط: أي رقم > 5 غير موجود في السياق."""
    if not answer or not chunks:
        return False
    context_text = " ".join(chunks).lower()
    numbers = re.findall(r'\d+', answer)
    for num in numbers:
        if int(num) > 5 and num not in context_text:
            return True
    return False

def log_interaction(
    log_path: str,
    interaction_id: str,
    timestamp: datetime,
    query: str,
    response: str,
    latency_ms: float,
    error: int,
    intent: str,
    chunks_retrieved: int,
    retrieval_quality: float,
    hallucination_flag: int,
    response_length: int,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    user_satisfaction: Optional[int] = None
) -> None:
    """إضافة صف جديد إلى ملف السجل."""
    short_query = query[:500]
    short_response = response[:500]

    row = [
        interaction_id,
        timestamp.isoformat(),
        short_query,
        short_response,
        f"{latency_ms:.2f}",
        error,
        intent,
        chunks_retrieved,
        retrieval_quality,
        hallucination_flag,
        response_length,
        prompt_tokens,
        completion_tokens,
        total_tokens,
        user_satisfaction if user_satisfaction is not None else ""
    ]
    with _write_lock:
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

def update_user_satisfaction(log_path: str, interaction_id: str, satisfaction: int) -> bool:
    """تحديث حقل رضا المستخدم لتفاعل محدد. يُرجع True إذا نجح."""
    with _write_lock:
        # قراءة الملف بالكامل
        if not os.path.exists(log_path):
            return False
        with open(log_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if len(rows) < 2:
            return False
        header = rows[0]
        # البحث عن عمود user_satisfaction و interaction_id
        try:
            id_col = header.index("interaction_id")
            sat_col = header.index("user_satisfaction")
        except ValueError:
            return False
        updated = False
        for i in range(1, len(rows)):
            if rows[i][id_col] == interaction_id:
                rows[i][sat_col] = str(satisfaction)
                updated = True
                break
        if updated:
            with open(log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(rows)
        return updated

def get_dashboard_metrics(log_path: str) -> dict:
    """حساب إحصائيات مجمعة من ملف السجل للداشبورد."""
    if not os.path.exists(log_path):
        return {}
    with _write_lock:
        with open(log_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    if not rows:
        return {}

    total = len(rows)
    errors = sum(1 for r in rows if r.get("error") == "1")
    hallucinations = sum(1 for r in rows if r.get("hallucination_flag") == "1")
    rated = [r for r in rows if r.get("user_satisfaction") not in (None, "")]
    positive = sum(1 for r in rated if r.get("user_satisfaction") == "1")
    negative = sum(1 for r in rated if r.get("user_satisfaction") == "0")

    latencies = [float(r["latency_ms"]) for r in rows if r.get("latency_ms")]
    retrieval_qualities = [float(r["retrieval_quality"]) for r in rows if r.get("retrieval_quality")]
    response_lengths = [int(r["response_length"]) for r in rows if r.get("response_length")]
    total_tokens = sum(int(r["total_tokens"]) for r in rows if r.get("total_tokens"))

    intents = {}
    for r in rows:
        intent = r.get("intent", "unknown")
        intents[intent] = intents.get(intent, 0) + 1

    return {
        "total_interactions": total,
        "error_rate": round((errors / total) * 100, 2) if total else 0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "avg_retrieval_quality": round(sum(retrieval_qualities) / len(retrieval_qualities), 2) if retrieval_qualities else 0,
        "hallucination_rate": round((hallucinations / total) * 100, 2) if total else 0,
        "user_satisfaction_rate": round((positive / len(rated)) * 100, 2) if rated else None,
        "total_positive": positive,
        "total_negative": negative,
        "avg_response_length": round(sum(response_lengths) / len(response_lengths), 2) if response_lengths else 0,
        "total_tokens_used": total_tokens,
        "intents_distribution": intents,
        "last_updated": datetime.now().isoformat()
    }