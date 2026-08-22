#!/usr/bin/env python3
"""GAJE Helix — Quantum Embedding Compression CLI.

Compresses a dense embedding table (e.g. from an LLM vocab) into a Quantum Superposition Codebook (.qemb)
with 8,192 canonical meta-tokens and m=4 sparse projections.
"""

import os
import sys
import time
import argparse
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.processing.quantum_codebook import (
    QuantumCodebook,
    QuantumEmbeddingTable,
)


def compress_embeddings_cli(
    input_path: str,
    output_path: str,
    num_meta_tokens: int = 8192,
    m: int = 4,
    iterations: int = 15,
):
    print("=" * 80)
    print("🧬 GAJE HELIX — QUANTUM EMBEDDING COMPRESSION ENGINE")
    print(f"Meta-Tokens Cuánticos: {num_meta_tokens} | Superposiciones: m={m}")
    print("=" * 80)

    # 1. Cargar embeddings o generar sintéticos para benchmark
    if input_path and os.path.exists(input_path):
        print(f"📂 Cargando embeddings desde: {input_path}...")
        if input_path.endswith(".npy"):
            dense_emb = np.load(input_path).astype(np.float32)
        else:
            raise ValueError("Formato de entrada no soportado. Use .npy")
    else:
        print(f"⚡ Generando matriz de prueba estándar (151,643 tokens x 896 dim)...")
        np.random.seed(42)
        dense_emb = np.random.randn(151643, 896).astype(np.float32)

    num_tokens, dim = dense_emb.shape
    raw_size_mb = (num_tokens * dim * 4) / (1024 * 1024)
    print(f"📊 Dimensiones: {num_tokens} tokens x {dim} dim (Tamaño FP32: {raw_size_mb:.2f} MB)")

    # 2. Entrenar Codebook y Proyectar
    t0 = time.time()
    print(f"⏳ Ajustando Codebook Cuántico de {num_meta_tokens} estados ({iterations} iteraciones)...")
    codebook = QuantumCodebook(num_meta_tokens, dim)
    codebook.fit_from_embeddings(dense_emb, num_iterations=iterations, batch_size=8192)

    print(f"⏳ Proyectando {num_tokens} tokens a superposiciones cuánticas m={m}...")
    table = QuantumEmbeddingTable(codebook, num_tokens, m=m)
    for t_i in range(num_tokens):
        inds, amps = codebook.project_sparse(dense_emb[t_i], m=m)
        table.indices[t_i] = inds
        table.amplitudes[t_i] = [int(min(255, max(0, round(a * 255.0)))) for a in amps]

    elapsed = time.time() - t0

    # 3. Guardar .qemb
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    table.save_qemb(output_path)
    qemb_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    savings = (1.0 - (qemb_size_mb / raw_size_mb)) * 100.0

    # 4. Evaluar fidelidad de reconstrucción
    fidelity = codebook.evaluate_reconstruction_fidelity(dense_emb, m=m, sample_size=1000)

    print("\n" + "=" * 80)
    print("🏆 RESULTADOS DE LA COMPRESIÓN CUÁNTICA")
    print("=" * 80)
    print(f"• Archivo de salida: {output_path}")
    print(f"• Tamaño original FP32: {raw_size_mb:.2f} MB")
    print(f"• Tamaño comprimido .qemb: {qemb_size_mb:.2f} MB")
    print(f"• Ahorro de memoria RAM: {savings:.2f}% ({raw_size_mb / qemb_size_mb:.1f}x compresión)")
    print(f"• Fidelidad de Reconstrucción (CosSim): {fidelity:.4f}")
    print(f"• Tiempo total de procesamiento: {elapsed:.2f} s")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAJE Quantum Embedding Compression CLI")
    parser.add_argument("--input", type=str, default="", help="Ruta al archivo .npy de embeddings")
    parser.add_argument("--output", type=str, default="data/memory/quantum_vocab.qemb", help="Ruta de salida .qemb")
    parser.add_argument("--meta_tokens", type=int, default=8192, help="Número de meta-tokens canónicos")
    parser.add_argument("--m", type=int, default=4, help="Número de proyecciones por token")
    parser.add_argument("--iterations", type=int, default=10, help="Iteraciones de clustering")

    args = parser.parse_args()
    compress_embeddings_cli(
        input_path=args.input,
        output_path=args.output,
        num_meta_tokens=args.meta_tokens,
        m=args.m,
        iterations=args.iterations,
    )
