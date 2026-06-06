import os
import sys
import numpy as np
import argparse
import random
import time

# Asegurar uso de código local
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python")))

from gaje.nn.stabilized import GenomicLLM

def run_phase_3_1(gaje_path, context_lengths=[512, 1024, 2048]):
    print(f"🚀 Iniciando Fase 3.1: La Aguja en el Pajar (Needle in a Haystack)")
    print(f"[*] Organismo: {gaje_path}")
    print(f"[*] Contextos a probar: {context_lengths}")
    
    # 1. Cargar GAJE
    print("[~] Cargando modelo GAJE...")
    gaje_model = GenomicLLM.load_genomic(gaje_path)
    tokenizer = gaje_model.tokenizer
    
    needle = "El código secreto de la misión es GAJE-2026."
    question = "¿Cuál es el código secreto de la misión?"
    expected_answer = "GAJE-2026"
    
    # Dataset para el "pajar" (ruido irrelevante)
    haystack_file = "data/datasets/hybrid_polyglot_dataset.txt"
    if not os.path.exists(haystack_file):
        print(f"❌ Error: No se encontró el dataset de ruido en {haystack_file}")
        return
        
    with open(haystack_file, "r", encoding="utf-8") as f:
        haystack_text = f.read()
        
    results = []
    
    for length in context_lengths:
        print(f"\n--- Probando Contexto: {length} tokens ---")
        
        # Generar pajar de la longitud deseada
        # Aproximadamente 4 caracteres por token
        current_haystack = haystack_text[:length * 4] 
        haystack_tokens = tokenizer.encode(current_haystack, add_special_tokens=False)
        
        # Ajustar para tener exactamente 'length' tokens antes de la aguja
        haystack_tokens = haystack_tokens[:length]
        
        # Insertar la aguja en una posición aleatoria (o al medio para estrés uniforme)
        insertion_point = len(haystack_tokens) // 2
        needle_tokens = tokenizer.encode(needle, add_special_tokens=False)
        
        full_context_tokens = haystack_tokens[:insertion_point] + needle_tokens + haystack_tokens[insertion_point:]
        question_tokens = tokenizer.encode(question, add_special_tokens=False)
        
        # Inferencia con KV-Cache
        print(f"[~] Procesando pajar ({len(full_context_tokens)} tokens)...")
        start_time = time.time()
        
        # Limpiar caché y pre-llenar con el contexto
        gaje_model.rust_llm.clear_cache_py()
        for i, tid in enumerate(full_context_tokens):
            gaje_model.rust_llm.forward(tid, False)
            if (i+1) % 500 == 0:
                print(f"    [*] {i+1} tokens procesados...")
        
        # Procesar pregunta
        for tid in question_tokens[:-1]:
            gaje_model.rust_llm.forward(tid, False)
            
        # Generar respuesta
        print("[*] Generando respuesta: ", end="", flush=True)
        response_tokens = []
        # El último token de la pregunta dispara la primera generación
        next_logits = gaje_model.rust_llm.forward(question_tokens[-1], False)
        
        for _ in range(15): # Respuesta corta esperada
            # Greedy decoding para máxima certeza
            next_id = int(np.argmax(next_logits))
            if next_id == tokenizer.eos_token_id:
                break
            
            word = tokenizer.decode([next_id])
            print(word, end="", flush=True)
            response_tokens.append(next_id)
            
            # Siguiente logit
            next_logits = gaje_model.rust_llm.forward(next_id, False)
            
        full_response = tokenizer.decode(response_tokens)
        elapsed = time.time() - start_time
        
        # Validar si la aguja fue encontrada
        found = expected_answer.lower() in full_response.lower()
        results.append({
            "length": length,
            "found": found,
            "response": full_response,
            "time": elapsed
        })
        
        print(f"\n[Result] Longitud {length}: {'✅ ENCONTRADA' if found else '❌ PERDIDA'}")

    print("\n" + "="*60)
    print(f"📊 RESULTADO FINAL FASE 3.1")
    for r in results:
        status = '✅' if r['found'] else '❌'
        print(f"  - Contexto {r['length']:>4}: {status} | Tiempo: {r['time']:.2f}s | Resp: {r['response'][:30]}...")
    
    score = sum(1 for r in results if r['found']) / len(results) * 100
    print(f"\n  - Tasa de Recuperación: {score:.2f}%")
    print(f"  - KPI Meta (>80%):      {'✅ PASSED' if score >= 80 else '❌ FAILED'}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fase 3.1: Stress Test de KV-Cache")
    parser.add_argument("--gaje", type=str, default="models/checkpoints/gold_embryo.gaje", help="Ruta al archivo .gaje")
    parser.add_argument("--lengths", type=str, default="256,512,1024", help="Longitudes de contexto separadas por comas")
    args = parser.parse_args()

    context_lengths = [int(l) for l in args.lengths.split(",")]
    run_phase_3_1(args.gaje, context_lengths=context_lengths)
