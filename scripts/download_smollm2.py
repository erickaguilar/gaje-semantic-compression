import os
from huggingface_hub import hf_hub_download

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "data", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def download_smollm2():
    target_path = os.path.join(MODELS_DIR, "smollm2-135m-instruct-fp16.gguf")
    if os.path.exists(target_path):
        print(f"✅ SmolLM2-135M FP16 GGUF ya existe en: {target_path}")
        return target_path

    print(
        "📥 Descargando SmolLM2-135M-Instruct FP16 GGUF (270 MB) desde HuggingFace..."
    )
    downloaded_file = hf_hub_download(
        repo_id="bartowski/SmolLM2-135M-Instruct-GGUF",
        filename="SmolLM2-135M-Instruct-f16.gguf",
        local_dir=MODELS_DIR,
    )
    # Renombrar para consistencia
    os.rename(downloaded_file, target_path)
    print(f"✅ SmolLM2 descargado exitosamente en: {target_path}")
    return target_path


if __name__ == "__main__":
    download_smollm2()
