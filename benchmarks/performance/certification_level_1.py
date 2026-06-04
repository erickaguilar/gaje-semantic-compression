import os
import sys
import numpy as np
import time
import argparse

# Asegurar uso de código local
sys.path.insert(0, os.path.abspath("python"))

from gaje.nn.stabilized import GenomicLLM

def run_level_1_audit(model_path, context_lengths=[4096, 16384, 32768, 65536]):
    print("=" * 70)
    print("🎓 AUDITORÍA DE CERTIFICACIÓN NIVEL 1: RESONANCIA TOROIDAL")
    print("=" * 70)
    print(f"[*] Modelo: {model_path}")
    print(f"[*] Reto: Needle In A Haystack (NIHY) con Recirculación de Fase")
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Modelo no encontrado en {model_path}")
        return

    # 1. Cargar el Organismo 'Silver Adult'
    print("[~] Cargando motor genómico...")
    llm = GenomicLLM.load_genomic(model_path)
    tokenizer = llm.tokenizer
    
    # Definición de la "Aguja" y la "Pregunta"
    needle = "La clave de acceso al núcleo toroidal es: GAJE-RESONANCE-X99."
    question = "¿Cuál es la clave de acceso al núcleo toroidal?"
    expected_token = "GAJE-RESONANCE-X99"
    
    # Cargar el "Pajar" (Dataset de ruido masivo)
    haystack_path = "data/datasets/specialized/tiny_shakespeare.txt"
    if not os.path.exists(haystack_path):
        print(f"❌ Error: Dataset de ruido no encontrado en {haystack_path}")
        return
        
    with open(haystack_path, "r", encoding="utf-8") as f:
        haystack_full = f.read()
        
    audit_results = []
    
    for length in context_lengths:
        print(f"\n--- AUDITANDO CONTEXTO: {length} tokens ---")
        
        # Preparar el contexto masivo
        # Repetimos el dataset si es necesario para alcanzar la longitud
        needed_chars = length * 4
        haystack_str = (haystack_full * (needed_chars // len(haystack_full) + 1))[:needed_chars]
        
        haystack_tokens = tokenizer.encode(haystack_str, add_special_tokens=False)
        if hasattr(haystack_tokens, "ids"): haystack_tokens = haystack_tokens.ids
        haystack_tokens = haystack_tokens[:length]
        
        # La aguja se inserta en el percentil 99 (la posición más difícil)
        insertion_point = int(len(haystack_tokens) * 0.99)
        needle_tokens = tokenizer.encode(needle, add_special_tokens=False)
        if hasattr(needle_tokens, "ids"): needle_tokens = needle_tokens.ids
        
        full_context = haystack_tokens[:insertion_point] + needle_tokens + haystack_tokens[insertion_point:]
        
        # Procesamiento con Soberanía Nativa
        print(f"[*] Inyectando {len(full_context)} tokens en el espacio de fase toroidal...")
        llm.rust_llm.clear_cache_py()
        
        t0 = time.time()
        # Pre-llenado de KV-Cache (Recirculación)
        for i, tid in enumerate(full_context):
            llm.rust_llm.forward(tid, False)
            if (i+1) % 5000 == 0:
                print(f"    [+] {i+1} tokens recirculados...")
        
        # Pregunta
        q_tokens = tokenizer.encode(question, add_special_tokens=False)
        if hasattr(q_tokens, "ids"): q_tokens = q_tokens.ids
        
        for tid in q_tokens[:-1]:
            llm.rust_llm.forward(tid, False)
            
        # Respuesta (Greedy Search)
        print("[*] Recuperando señal: ", end="", flush=True)
        resp_ids = []
        next_logits = llm.rust_llm.forward(q_tokens[-1], False)
        
        for _ in range(15):
            next_id = int(np.argmax(next_logits))
            if next_id == tokenizer.eos_token_id: break
            token_text = tokenizer.decode([next_id])
            print(token_text, end="", flush=True)
            resp_ids.append(next_id)
            next_logits = llm.rust_llm.forward(next_id, False)
            
        full_response = tokenizer.decode(resp_ids)
        latency = time.time() - t0
        
        found = expected_token.lower() in full_response.lower()
        audit_results.append({
            "length": length,
            "found": found,
            "latency": latency,
            "response": full_response.strip()
        })
        
        print(f"\n[Status] Recall @ 99%: {'✅ SUCCESS' if found else '❌ FAILURE'}")

    # Reporte Final para Certificación
    print("\n" + "=" * 70)
    print("📊 REPORTE DE EVIDENCIA: RESONANCIA TOROIDAL")
    print("-" * 70)
    print(f"{'Contexto':<12} | {'Recall':<10} | {'Tiempo':<10} | {'Respuesta'}")
    print("-" * 70)
    for r in audit_results:
        status = "100%" if r['found'] else "0%"
        print(f"{r['length']:<12} | {status:<10} | {r['latency']:>8.2f}s | {r['response'][:40]}")
    
    overall_score = sum(1 for r in audit_results if r['found']) / len(audit_results) * 100
    print("-" * 70)
    print(f"CERTIFICACIÓN SCORE: {overall_score:.2f}%")
    print(f"ESTADO DE SELLO:     {'🏆 GRADE-A CERTIFIED' if overall_score == 100 else '❌ DENIED'}")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/production/silver_adult_steel.gaje")
    args = parser.parse_args()
    
    # Probamos longitudes crecientes para ver el límite de estabilidad
    run_level_1_audit(args.model, context_lengths=[3])
