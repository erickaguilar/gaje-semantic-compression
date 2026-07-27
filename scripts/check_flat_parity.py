import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.core import _impl as dna_semantic_compression

def check_parity():
    db_path = os.path.join(PROJECT_ROOT, "models", "production", "qwen2_0_5b_4bit.gaje")
    flat_path = os.path.join(PROJECT_ROOT, "models", "production", "qwen2_0_5b_4bit.gaje.flat")

    print(f"[*] Cargando modelo DB: {db_path}")
    db_loader = dna_semantic_compression.NativeLoader(db_path)
    llm_db = db_loader.py_load_llm()

    print(f"[*] Cargando modelo Flat Mmap: {flat_path}")
    llm_flat = dna_semantic_compression.load_genomic_auto(flat_path)

    # 2. Comparar Forward Pass Token
    llm_db.set_k_wta_ratio(0.0)
    llm_flat.set_k_wta_ratio(0.0)

    test_token_id = 4929 # ' La'
    logits_db = llm_db.forward(test_token_id, True)
    logits_flat = llm_flat.forward(test_token_id, True)

    cos_logits = np.dot(logits_db, logits_flat) / (np.linalg.norm(logits_db) * np.linalg.norm(logits_flat))
    top1_db = np.argmax(logits_db)
    top1_flat = np.argmax(logits_flat)

    print(f"  └─ Logits Finales CosSim: {cos_logits:.6f}")
    print(f"  └─ Top-1 DB: {top1_db} | Top-1 Flat: {top1_flat}")

if __name__ == "__main__":
    check_parity()
