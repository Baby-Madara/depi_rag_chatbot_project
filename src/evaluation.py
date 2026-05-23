import os
import time
import re
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import numpy as np

_embedding_model = None

def get_embedding(text):
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return _embedding_model.encode(text)

def calculate_cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))

def is_mostly_english(text):
    en_chars = len(re.findall(r'[a-zA-Z]', text))
    ar_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    return en_chars > ar_chars

def tokenize_text(text):
    text = text.lower()
    text = re.sub(r'([?.!,¿؛،])', r' \1 ', text)
    return text.split()

def load_golden_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "golden_dataset.xlsx")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Golden dataset not found at {dataset_path}")
    df = pd.read_excel(dataset_path)
    df = df.fillna("")
    return df.to_dict(orient="records")

def run_system_evaluation(rag_chain_wrapper, custom_dataset=None):
    if custom_dataset is not None:
        golden_dataset = custom_dataset
    else:
        golden_dataset = load_golden_dataset()

    total_bleu = 0
    total_rouge = 0
    total_similarity = 0
    hit_count = 0
    hallucination_count = 0
    total_latency = 0

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    # استخدام method7 بدلاً من method1 – أفضل للجمل القصيرة
    smoothing = SmoothingFunction()

    for case in golden_dataset:
        start_time = time.time()
        generated_answer, retrieved_chunks, _ = rag_chain_wrapper(case)
        latency = time.time() - start_time
        total_latency += latency

        eval_expected = str(case["expected_answer"]).strip()
        eval_generated = generated_answer.strip()



        ref_tokens = tokenize_text(eval_expected)
        gen_tokens = tokenize_text(eval_generated)

        # BLEU مع method7 (سلسة وتناسب الجمل القصيرة)
        bleu = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smoothing.method7)
        total_bleu += bleu

        # ROUGE
        rouge_scores = scorer.score(eval_expected, eval_generated)
        total_rouge += rouge_scores['rougeL'].fmeasure

        # Semantic similarity
        vec_ref = get_embedding(eval_expected)
        vec_gen = get_embedding(eval_generated)
        sim = calculate_cosine_similarity(vec_ref, vec_gen)
        total_similarity += sim

        # Hit Rate
        context_text = " ".join(retrieved_chunks).lower()
        if len(retrieved_chunks) > 0 and sim > 0.35:
            hit_count += 1

        # Hallucination
        has_hallucination = False
        numbers = re.findall(r'\d+', generated_answer)
        for num in numbers:
            if int(num) > 5 and num not in context_text and num not in str(case["query"]):
                has_hallucination = True
                break
        if has_hallucination:
            hallucination_count += 1

    num_cases = len(golden_dataset)
    if num_cases == 0:
        return {"bleu": 0, "rouge": 0, "hit_rate": 0, "latency": 0, "semantic_similarity": 0, "hallucination_rate": 0}

    return {
        "bleu": round((total_bleu / num_cases) * 100, 2),
        "rouge": round((total_rouge / num_cases) * 100, 2),
        "hit_rate": round((hit_count / num_cases) * 100, 2),
        "latency": round(total_latency / num_cases, 3),
        "semantic_similarity": round((total_similarity / num_cases) * 100, 2),
        "hallucination_rate": round((hallucination_count / num_cases) * 100, 2)
    }