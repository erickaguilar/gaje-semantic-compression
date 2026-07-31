import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.core import _impl as dna_semantic_compression  # noqa: E402


def compare():
    db_path = os.path.join(PROJECT_ROOT, "models", "production", "qwen2_0_5b_4bit.gaje")
    flat_path = os.path.join(
        PROJECT_ROOT, "models", "production", "qwen2_0_5b_4bit.gaje.flat"
    )

    prompt = "¿Cuál es la capital de Francia?"

    print("=== 1. MODELO REDB DATABASE (.gaje) ===")
    loader_db = dna_semantic_compression.NativeLoader(db_path)
    llm_db = loader_db.py_load_llm()
    llm_db.set_k_wta_ratio(0.0)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
    input_ids = tok.encode(prompt, add_special_tokens=False)

    gen_db = llm_db.generate_native_py(input_ids, 20, 0.3, 1.1, [2, 151643, 151645])
    print("Tokens DB:", gen_db)
    print("Texto DB:", tok.decode(gen_db))

    print("\n=== 2. MODELO FLAT MMAP (.gaje.flat) ===")
    llm_flat = dna_semantic_compression.load_genomic_auto(flat_path)
    llm_flat.set_k_wta_ratio(0.0)

    gen_flat = llm_flat.generate_native_py(input_ids, 20, 0.3, 1.1, [2, 151643, 151645])
    print("Tokens Flat:", gen_flat)
    print("Texto Flat:", tok.decode(gen_flat))


if __name__ == "__main__":
    compare()
