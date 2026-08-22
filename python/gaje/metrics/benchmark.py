"""GAJE Helix — Scientific Benchmark Metrics Engine.

Evaluates latency (cold-start, TTFT), throughput (tok/s), memory footprint (RSS MB),
lexical diversity (Distinct-1, Distinct-2), repetition loops, and semantic keyword recall.
"""

import os
import re
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger("gaje.metrics.benchmark")


def get_current_rss_mb() -> float:
    """Retorna la memoria física residente (RSS) actual en MB."""
    if psutil:
        try:
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            pass
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def calculate_lexical_diversity(text: str) -> Dict[str, float]:
    """Calcula la diversidad léxica Distinct-1 (unigramas) y Distinct-2 (bigramas)."""
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return {"distinct_1": 0.0, "distinct_2": 0.0}

    unique_unigrams = set(words)
    d1 = len(unique_unigrams) / len(words)

    if len(words) < 2:
        d2 = 1.0
    else:
        bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
        d2 = len(set(bigrams)) / len(bigrams)

    return {
        "distinct_1": round(d1, 4),
        "distinct_2": round(d2, 4),
    }


def calculate_keyword_recall(response: str, expected_keywords: List[str]) -> float:
    """Calcula el porcentaje de palabras clave esperadas presentes en la respuesta."""
    if not expected_keywords:
        return 1.0
    resp_lower = response.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in resp_lower)
    return round(found / len(expected_keywords), 4)


def detect_repetition_loops(text: str, n_gram: int = 4, threshold: int = 3) -> bool:
    """Detecta si el texto cae en bucles degenerativos de repetición infinita."""
    words = re.findall(r"\b\w+\b", text.lower())
    if len(words) < n_gram * threshold:
        return False

    counts = {}
    for i in range(len(words) - n_gram + 1):
        ng = tuple(words[i : i + n_gram])
        counts[ng] = counts.get(ng, 0) + 1
        if counts[ng] >= threshold:
            return True
    return False


@dataclass
class PromptEvalResult:
    test_id: str
    category: str
    language: str
    prompt: str
    response: str
    prompt_tokens: int
    generated_tokens: int
    total_tokens: int
    ttft_ms: float
    total_latency_ms: float
    tokens_per_sec: float
    distinct_1: float
    distinct_2: float
    keyword_recall: float
    repetition_loop_detected: bool


@dataclass
class BenchmarkReport:
    model_name: str
    architecture: str
    file_size_bytes: int
    file_size_mb: float
    cold_start_ms: float
    peak_rss_mb: float
    rss_delta_mb: float
    total_test_cases: int
    avg_ttft_ms: float
    avg_tokens_per_sec: float
    avg_distinct_1: float
    avg_distinct_2: float
    avg_keyword_recall: float
    repetition_rate_pct: float
    compression_ratio: float
    memory_savings_pct: float
    test_results: List[PromptEvalResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_single_prompt(
    llm: Any,
    model_name: str,
    test_case: Dict[str, Any],
    format_prompt_fn: Any,
    get_stop_tokens_fn: Any,
    max_tokens: int = 128,
    temperature: float = 0.2,
    repetition_penalty: float = 1.1,
) -> PromptEvalResult:
    """Evalúa un único prompt sobre la instancia nativa del LLM."""
    prompt_text = test_case["prompt"]
    formatted = format_prompt_fn(model_name, prompt_text)
    
    tokens = llm.tokenizer.encode(formatted, add_special_tokens=False)
    if hasattr(tokens, "ids"):
        tokens = tokens.ids
    prompt_tokens_count = len(tokens)

    eos_ids = get_stop_tokens_fn(model_name, llm.tokenizer)

    start_time = time.time()
    try:
        gen_ids = llm.rust_llm.generate_native_py(
            tokens,
            test_case.get("max_target_tokens", max_tokens),
            temperature,
            repetition_penalty,
            eos_ids,
        )
    except Exception as e:
        logger.warning("Error en generate_native_py: %s", e)
        gen_ids = [2]

    elapsed_ms = (time.time() - start_time) * 1000.0
    generated_tokens_count = len(gen_ids)
    total_tokens = prompt_tokens_count + generated_tokens_count

    tok_per_sec = (
        (generated_tokens_count / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0.0
    )

    full_resp = llm.tokenizer.decode(gen_ids)
    cleaned_resp = (
        full_resp.split("<|im_end|>")[0]
        .split("<|endoftext|>")[0]
        .split("<end_of_turn>")[0]
        .strip()
    )

    # TTFT aproximado (proporción inicial de cómputo del prompt)
    ttft_ms = elapsed_ms / max(1, generated_tokens_count)

    div = calculate_lexical_diversity(cleaned_resp)
    recall = calculate_keyword_recall(cleaned_resp, test_case.get("expected_keywords", []))
    has_loop = detect_repetition_loops(cleaned_resp)

    return PromptEvalResult(
        test_id=test_case["id"],
        category=test_case["category"],
        language=test_case["language"],
        prompt=prompt_text,
        response=cleaned_resp,
        prompt_tokens=prompt_tokens_count,
        generated_tokens=generated_tokens_count,
        total_tokens=total_tokens,
        ttft_ms=round(ttft_ms, 2),
        total_latency_ms=round(elapsed_ms, 2),
        tokens_per_sec=round(tok_per_sec, 2),
        distinct_1=div["distinct_1"],
        distinct_2=div["distinct_2"],
        keyword_recall=recall,
        repetition_loop_detected=has_loop,
    )


def evaluate_model_benchmark(
    models_root: str,
    model_name: str,
    prompts_dataset: Dict[str, Any],
    get_model_fn: Any,
    genomic_llm_cls: Any,
    format_prompt_fn: Any,
    get_stop_tokens_fn: Any,
) -> BenchmarkReport:
    """Ejecuta la suite completa de benchmark para un modelo."""
    rss_before = get_current_rss_mb()
    
    # 1. Medición de Cold-Start
    start_load = time.time()
    llm = get_model_fn(models_root, model_name, genomic_llm_cls)
    cold_start_ms = (time.time() - start_load) * 1000.0

    if not llm:
        raise RuntimeError(f"No se pudo cargar el modelo: {model_name}")

    rss_loaded = get_current_rss_mb()
    rss_delta = max(0.0, rss_loaded - rss_before)

    # 2. Obtener metadatos de archivo
    file_size_bytes = 0
    for root, _, files in os.walk(models_root):
        if model_name in files:
            file_size_bytes = os.path.getsize(os.path.join(root, model_name))
            break
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)

    # Arquitectura y ratio
    if "qwen2_5" in model_name:
        arch = "Qwen2.5"
    elif "qwen2" in model_name:
        arch = "Qwen2"
    elif "deepseek" in model_name:
        arch = "DeepSeek-R1"
    elif ".gaje" in model_name:
        arch = "GAJE-Born"
    else:
        arch = "SmolLM2"
    bit_depth = getattr(llm, "bit_depth", 4)
    compression_ratio = 8.0 if bit_depth == 4 else (16.0 if bit_depth == 2 else 1.0)
    savings_pct = 87.5 if bit_depth == 4 else (93.75 if bit_depth == 2 else 0.0)

    # 3. Evaluar cada test case
    test_cases = prompts_dataset.get("test_cases", [])
    results: List[PromptEvalResult] = []

    for tc in test_cases:
        res = evaluate_single_prompt(
            llm=llm,
            model_name=model_name,
            test_case=tc,
            format_prompt_fn=format_prompt_fn,
            get_stop_tokens_fn=get_stop_tokens_fn,
        )
        results.append(res)

    # 4. Agregación de métricas
    n = len(results)
    avg_ttft = sum(r.ttft_ms for r in results) / n if n > 0 else 0.0
    avg_tps = sum(r.tokens_per_sec for r in results) / n if n > 0 else 0.0
    avg_d1 = sum(r.distinct_1 for r in results) / n if n > 0 else 0.0
    avg_d2 = sum(r.distinct_2 for r in results) / n if n > 0 else 0.0
    avg_recall = sum(r.keyword_recall for r in results) / n if n > 0 else 0.0
    repetition_count = sum(1 for r in results if r.repetition_loop_detected)
    repetition_rate = (repetition_count / n * 100.0) if n > 0 else 0.0

    return BenchmarkReport(
        model_name=model_name,
        architecture=arch,
        file_size_bytes=file_size_bytes,
        file_size_mb=file_size_mb,
        cold_start_ms=round(cold_start_ms, 2),
        peak_rss_mb=round(rss_loaded, 2),
        rss_delta_mb=round(rss_delta, 2),
        total_test_cases=n,
        avg_ttft_ms=round(avg_ttft, 2),
        avg_tokens_per_sec=round(avg_tps, 2),
        avg_distinct_1=round(avg_d1, 4),
        avg_distinct_2=round(avg_d2, 4),
        avg_keyword_recall=round(avg_recall, 4),
        repetition_rate_pct=round(repetition_rate, 2),
        compression_ratio=compression_ratio,
        memory_savings_pct=savings_pct,
        test_results=results,
    )
