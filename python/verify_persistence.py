import os
import numpy as np
import dna_semantic_compression
import time
from python.genomize_llm import GenomicLLM

def verify_persistence(gaje_dir):
    print(f"🧬 Verificando Persistencia Genómica: {gaje_dir}")
    print("-" * 60)
    
    # 1. Cargar modelo desde disco (Sin GGUF original)
    start_load = time.perf_counter()
    model = GenomicLLM(gaje_dir, load_genomic=True)
    end_load = time.perf_counter()
    
    print(f"[*] Modelo cargado en {(end_load - start_load)*1000:.2f} ms")
    print(f"[*] Número de bloques detectados: {len(model.blocks)}")
    
    # 2. Realizar inferencia
    prompt = "El protocolo GAJE es un sistema"
    print(f"\n🚀 Probando generación desde el modelo completo (24 bloques):")
    model.generate(prompt, max_new_tokens=100, temperature=0.7)
    print("\n" + "-" * 60)
    print("✅ Verificación completada.")

if __name__ == "__main__":
    verify_persistence("gaje_qwen2_full_v1")
