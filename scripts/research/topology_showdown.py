import os
import sys
import numpy as np
import argparse
import time
from tqdm import tqdm

# Asegurar uso de código local
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python")))

from gaje.nn.stabilized import GenomicLLM

def calculate_ppl_showdown(model, text, tokenizer, max_length=128):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not tokens: return None
    tokens = tokens[:max_length]
    
    # Inferencia nativa
    logits_seq = []
    for tid in tokens:
        # El forward devuelve [vocab_size]
        logits = model.rust_llm.forward(tid, False)
        logits_seq.append(logits)
        
    logits_seq = logits_seq[:-1]
    target_tokens = tokens[1:]
    
    log_probs = []
    for i, target_id in enumerate(target_tokens):
        logits = logits_seq[i]
        # Softmax manual
        e_x = np.exp(logits - np.max(logits))
        probs = e_x / e_x.sum()
        p = np.clip(probs[target_id], 1e-10, 1.0)
        log_probs.append(np.log(p))
    
    return np.exp(-np.mean(log_probs)) if log_probs else None

def run_showdown(gaje_path, topo_rust_path, topo_es_path):
    print(f"🏔️  INICIANDO EL DUELO: The Topology Showdown (Fase 4.0)")
    print(f"[*] Modelo Base: {gaje_path}")
    
    # 1. Cargar Modelos
    model = GenomicLLM.load_genomic(gaje_path)
    tokenizer = model.tokenizer
    
    datasets = {
        "Rust": "data/datasets/expert_rust.txt",
        "Español": "data/datasets/coherence_es.txt"
    }
    
    results = {}

    for label, path in datasets.items():
        print(f"\n--- Evaluando Dominio: {label} ---")
        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if len(l.strip()) > 20][:10]

        # A. Medir Línea Base
        print(f"[~] Calculando PPL Línea Base (Solo ADN)...")
        model.rust_llm.clear_cache_py()
        ppls_base = []
        for line in tqdm(lines, desc="Base"):
            res = calculate_ppl_showdown(model, line, tokenizer)
            if res: ppls_base.append(res)
        avg_base = np.mean(ppls_base)
        
        # B. Inyectar Topología Correspondiente
        topo_path = topo_rust_path if label == "Rust" else topo_es_path
        print(f"[~] Inyectando Topología Relacional desde {os.path.basename(topo_path)}...")
        model.rust_llm.load_topology(topo_path)
        
        # C. Medir Aumento
        print(f"[~] Calculando PPL Aumentada (ADN + Grafo)...")
        model.rust_llm.clear_cache_py()
        ppls_topo = []
        for line in tqdm(lines, desc="Topología"):
            res = calculate_ppl_showdown(model, line, tokenizer)
            if res: ppls_topo.append(res)
        avg_topo = np.mean(ppls_topo)
        
        # Guardar resultados
        improvement = ((avg_base - avg_topo) / avg_base) * 100
        results[label] = {
            "base": avg_base,
            "topo": avg_topo,
            "gain": improvement
        }

    # 📊 Reporte Final de Duelo
    print("\n" + "="*65)
    print(f"{'DOMINIO':<15} | {'PPL BASE':<12} | {'PPL GRAFO':<12} | {'MEJORA %':<10}")
    print("-" * 65)
    for label, data in results.items():
        print(f"{label:<15} | {data['base']:>12.2f} | {data['topo']:>12.2f} | {data['gain']:>9.2f}%")
    
    # KPI Final
    avg_gain = np.mean([d['gain'] for d in results.values()])
    print("-" * 65)
    print(f"📊 IMPACTO TOPOLÓGICO PROMEDIO: {avg_gain:.2f}%")
    print(f"HITO DE VIABILIDAD: {'✅ SUPERADO' if avg_gain > 20 else '❌ EN DUDA'}")
    print("="*65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paso 3: The Topology Showdown")
    parser.add_argument("--gaje", type=str, default="models/checkpoints/gold_embryo.gaje")
    parser.add_argument("--topo_rust", type=str, default="models/core/topology_rust.json")
    parser.add_argument("--topo_es", type=str, default="models/core/topology_es.json")
    args = parser.parse_args()

    run_showdown(args.gaje, args.topo_rust, args.topo_es)
