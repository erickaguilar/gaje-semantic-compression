"""GAJE-Flow Model Manager.

Handles recursive model discovery, thread-safe lazy loading, memory caching,
and model unloading to optimize RAM consumption.
"""

from datetime import datetime
import gc
import os
import threading

loaded_models = {}
model_lock = threading.Lock()


def find_model_path(models_root: str, model_name: str) -> str:
    """Recursively search for a model file (.gaje or .flat) inside models_root."""
    if not os.path.exists(models_root):
        return None

    for root, _, files in os.walk(models_root):
        if model_name in files:
            return os.path.join(root, model_name)
    return None


def get_model(models_root: str, model_name: str, GenomicLLM):
    """Thread-safe retrieval of a loaded GenomicLLM model instance."""
    with model_lock:
        if model_name in loaded_models:
            return loaded_models[model_name]

        model_path = find_model_path(models_root, model_name)
        if not model_path:
            print(
                f"❌ No se encontró el archivo de modelo '{model_name}' en {models_root}"
            )
            return None

        print(f"🧬 Cargando modelo real: {model_path}")
        try:
            # Unload previous models to free RAM if memory usage is tight
            if len(loaded_models) > 2:
                print("🧹 Liberando modelos inactivos de la memoria RAM...")
                loaded_models.clear()
                gc.collect()

            llm = GenomicLLM.load_genomic(os.path.abspath(model_path))
            llm.rust_llm.set_k_wta_ratio(0.0)
            loaded_models[model_name] = llm
            return llm
        except Exception as e:
            import traceback

            print(f"❌ Error cargando modelo {model_name}: {e}")
            traceback.print_exc()
            return None


def list_available_models(models_root: str) -> list:
    """List all available .gaje and .flat models with modification dates."""
    models = []
    seen_models = set()

    if os.path.exists(models_root):
        for root, _, files in os.walk(models_root):
            for f in files:
                if (
                    f.endswith(".gaje") or f.endswith(".flat")
                ) and f not in seen_models:
                    fpath = os.path.join(root, f)
                    mtime = os.path.getmtime(fpath)
                    date_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
                    models.append({"name": f, "date": date_str})
                    seen_models.add(f)

    return models
