"""GAJE-Flow Model Manager.

Handles recursive model discovery, thread-safe lazy loading, memory caching,
and model unloading to optimize RAM consumption.
"""

from datetime import datetime
import gc
import logging
import os
import threading

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger("gaje-web-ui.model-manager")

loaded_models = {}
loaded_ram_mb = {}
model_lock = threading.Lock()


def find_model_path(models_root: str, model_name: str) -> str:
    """Recursively search for a model file (.gaje or .flat) inside models_root.

    Seguridad (Fase 2.5): valida que `model_name` sea un simple nombre de archivo,
    no una ruta, para evitar path traversal.
    """
    if not model_name:
        return None

    # Bloquear path traversal: solo se acepta el nombre base (sin separadores)
    base = os.path.basename(model_name)
    if (
        base != model_name
        or ".." in model_name
        or "/" in model_name
        or "\\" in model_name
    ):
        logger.warning(
            "Nombre de modelo inválido (posible path traversal): %r", model_name
        )
        return None

    if not os.path.exists(models_root):
        return None

    for root, _, files in os.walk(models_root):
        if model_name in files:
            return os.path.join(root, model_name)
    return None


def process_rss_mb() -> float:
    """Resident memory (RSS) del proceso actual, en MB."""
    if psutil is not None:
        try:
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            pass
    return 0.0


def get_model(models_root: str, model_name: str, GenomicLLM):
    """Thread-safe retrieval of a loaded GenomicLLM model instance."""
    with model_lock:
        if model_name in loaded_models:
            return loaded_models[model_name]

        model_path = find_model_path(models_root, model_name)
        if not model_path:
            logger.error(
                "No se encontró el archivo de modelo '%s' en %s",
                model_name,
                models_root,
            )
            return None

        logger.info("Cargando modelo real: %s", model_path)
        try:
            # Strictly keep only ONE active model in RAM at any time
            if loaded_models:
                logger.info("Liberando modelo previo de la memoria RAM...")
                loaded_models.clear()
                gc.collect()

            rss_before = process_rss_mb()
            llm = GenomicLLM.load_genomic(os.path.abspath(model_path))
            llm.rust_llm.set_k_wta_ratio(0.0)
            rss_after = process_rss_mb()
            ram_mb = max(0.0, rss_after - rss_before)
            loaded_models[model_name] = llm
            loaded_ram_mb[model_name] = ram_mb
            logger.info("Modelo %s cargado; RAM residente ~%.1f MB", model_name, ram_mb)
            return llm
        except Exception as e:
            logger.error("Error cargando modelo %s: %s", model_name, e, exc_info=True)
            return None


def list_available_models(models_root: str) -> list:
    """List all certified .flat models from models/production/."""
    models = []
    seen_models = set()
    production_root = os.path.join(models_root, "production")

    if os.path.exists(production_root):
        for root, _, files in os.walk(production_root):
            for f in files:
                if f.endswith(".flat") and f not in seen_models:
                    fpath = os.path.join(root, f)
                    mtime = os.path.getmtime(fpath)
                    date_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
                    size_bytes = os.path.getsize(fpath)
                    models.append(
                        {
                            "name": f,
                            "date": date_str,
                            "size_bytes": size_bytes,
                            "ram_mb": loaded_ram_mb.get(f, 0.0),
                        }
                    )
                    seen_models.add(f)

    return models
