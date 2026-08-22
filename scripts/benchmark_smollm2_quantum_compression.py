#!/usr/bin/env python3
"""GAJE Helix — Real-world Quantum Compression on SmolLM2-135M Embeddings.

Extracts the real 49,152 x 576 FP32 embedding matrix from models/production/smollm2_135m.flat,
compresses it into a Quantum Codebook (.qemb), and benchmarks fidelity and memory reduction.
"""

import os
import sys
import time
import json
import struct
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.processing.quantum_codebook import (
    QuantumCodebook,
    QuantumEmbeddingTable,
)


def run_smollm2_quantum_compression():
    model_path = "models/production/smollm2_135m.flat"
    output_qemb = "data/memory/smollm2_embeddings.qemb"
    os.makedirs("data/memory", exist_ok=True)

    print("=" * 80)
    print("🧬 GAJE HELIX — COMPRESIÓN CUÁNTICA DE EMBEDDINGS EN PRODUCCIÓN")
    print(f"Modelo: {model_path}")
    print("=" * 80)

    # 1. Extraer token_embd real de smollm2_135m.flat
    print("📂 Leyendo cabecera y tabla de tensores de smollm2_135m.flat...")
    with open(model_path, "rb") as f:
        header = f.read(4096)
        magic, ver, flags, num_tensors, meta_len, dir_len, w_off, w_len = struct.unpack_from("<4sIIIQQQQ", header, 0)
        meta = json.loads(f.read(meta_len).decode("utf-8"))
        directory = json.loads(f.read(dir_len).decode("utf-8"))

        emb_info = None
        for item in directory:
            if item["name"] == "token_embd":
                emb_info = item
                break

        if not emb_info:
            raise RuntimeError("No se encontró el tensor token_embd en el modelo")

        vocab_size = emb_info["out_features"]
        n_embd = emb_info["in_features"]
        dna_off = emb_info["dna_off"]
        dna_len = emb_info["dna_len"]

        # Buscar offset absoluto
        f.seek(w_off + dna_off)
        raw_emb_bytes = f.read(dna_len)
        real_emb = np.frombuffer(raw_emb_bytes, dtype=np.float32).reshape(vocab_size, n_embd).copy()

    raw_mb = (vocab_size * n_embd * 4) / (1024 * 1024)
    print(f"📊 Tensor 'token_embd' extraído: {vocab_size} tokens x {n_embd} dim")
    print(f"   • Tamaño original en FP32: {raw_mb:.2f} MB ({len(raw_emb_bytes):,} bytes)")

    # 2. Compresión Cuántica (K=4096 / 8192, m=4)
    K = 4096
    m = 4
    print(f"\n⏳ Entrenando Quantum Codebook de {K} meta-tokens canónicos...")
    t0 = time.time()
    codebook = QuantumCodebook(num_meta_tokens=K, dim=n_embd)
    codebook.fit_from_embeddings(real_emb, num_iterations=12, batch_size=8192)
    fit_time = time.time() - t0
    print(f"   • Codebook ajustado en {fit_time:.2f} s")

    print(f"⏳ Proyectando los {vocab_size} tokens a superposiciones cuánticas m={m}...")
    t1 = time.time()
    table = QuantumEmbeddingTable(codebook, vocab_size, m=m)
    table.indices, table.amplitudes = codebook.project_batch(real_emb, m=m)
    proj_time = time.time() - t1
    print(f"   • Proyección completada en {proj_time:.2f} s")

    # 3. Guardar .qemb
    table.save_qemb(output_qemb)
    qemb_bytes = os.path.getsize(output_qemb)
    qemb_mb = qemb_bytes / (1024 * 1024)
    savings = (1.0 - (qemb_bytes / len(raw_emb_bytes))) * 100.0

    # 4. Medir fidelidad de reconstrucción en tokens reales
    sample_size = 2000
    fidelity = codebook.evaluate_reconstruction_fidelity(real_emb, m=m, sample_size=sample_size)

    # 5. Benchmark de latencia de lookup
    t2 = time.time()
    for t_i in range(10000):
        _ = table.get_embedding(t_i % vocab_size)
    lookup_us = ((time.time() - t2) / 10000) * 1_000_000.0

    print("\n" + "=" * 80)
    print("🏆 RESULTADOS OFICIALES — COMPRESIÓN CUÁNTICA EN SMOL_LM2 135M")
    print("=" * 80)
    print(f"• Archivo generado: {output_qemb}")
    print(f"• Tamaño original en FP32:      {raw_mb:.2f} MB")
    print(f"• Tamaño comprimido (.qemb):    {qemb_mb:.2f} MB")
    print(f"• Reducción de Memoria RAM:     {savings:.2f}% ({raw_mb / qemb_mb:.1f}x más ligero)")
    print(f"• Fidelidad Real (CosSim):      {fidelity:.4f}")
    print(f"• Latencia de Lookup al Vuelo:  {lookup_us:.2f} µs por token")
    print("=" * 80)


if __name__ == "__main__":
    run_smollm2_quantum_compression()
