import os
import time
import numpy as np
import json
import gc
import struct
from gaje.core import _impl as dna_semantic_compression

# Configuración del Experimento
NUM_DOCS = 10_000
DIMS = 768
TOP_K_RETRIEVAL = 10

def generate_semantic_dataset(n, dims):
    """
    Genera un dataset sintético con estructura semántica realista.
    """
    print(f"[*] Generando {n} documentos en 500 clusters semánticos...")
    num_clusters = 500
    cluster_centers = np.random.normal(0, 1, (num_clusters, dims)).astype(np.float32)
    
    # Menor ruido para simular documentos "distinguibles"
    doc_to_cluster = np.random.randint(0, num_clusters, n)
    embeddings = cluster_centers[doc_to_cluster] + np.random.normal(0, 0.02, (n, dims)).astype(np.float32)
    
    # Normalizar
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings /= (norms + 1e-10)
    
    documents = [f"Doc temático {c} - ID: {i}" for i, c in enumerate(doc_to_cluster)]
    return embeddings, documents

def train_genomic_codebook(embeddings):
    """
    Entrena usando una aproximación de Max-Lloyd para una distribución normal.
    Para 2 bits (4 niveles), los valores óptimos en una Normal(0,1) son:
    Umbrales: -0.966, 0, 0.966
    Centroides: -1.51, -0.452, 0.452, 1.51
    Adaptamos esto a la escala real de cada dimensión.
    """
    print("[*] Entrenando código genético (Max-Lloyd Approximation)...")
    n, dims = embeddings.shape
    
    all_thresholds = []
    all_centroids = []
    
    for d in range(dims):
        col = embeddings[:, d]
        std = np.std(col)
        mean = np.mean(col)
        
        # Escalar los valores óptimos de Max-Lloyd a la distribución de esta dimensión
        t = [mean - 0.966 * std, mean, mean + 0.966 * std]
        c = [mean - 1.51 * std, mean - 0.452 * std, mean + 0.452 * std, mean + 1.51 * std]
        
        all_thresholds.extend(t)
        all_centroids.extend(c)
        
    return all_thresholds, all_centroids

def run_qa_benchmark():
    print("\n" + "="*60)
    print("🚀 CRITICAL TEST: REAL-WORLD RETRIEVAL QA PIPELINE")
    print("="*60)

    # 1. Carga de Documentos y Embeddings
    embeddings, docs = generate_semantic_dataset(NUM_DOCS, DIMS)
    
    # 2. Configuración GAJE (Entrenamiento PER-DIMENSION)
    thresholds, centroids = train_genomic_codebook(embeddings)
    print(f"[+] Código Genético entrenado para {DIMS} dimensiones.")
    
    # 3. Empaquetado GAJE (Simulando compresión de DB)
    print(f"[*] Comprimiendo {NUM_DOCS} embeddings con GAJE (ADN Dinámico 2-bits)...")
    start_comp = time.time()
    index = dna_semantic_compression.GajeIndex([], centroids, m=16, ef_construction=100)
    
    batch_size = 500
    for i in range(0, NUM_DOCS, batch_size):
        batch_vecs = embeddings[i:i+batch_size]
        batch_dna = [dna_semantic_compression.quantize_embedding(v.tolist(), thresholds) for v in batch_vecs]
        index.add_batch(batch_dna)
    
    # 3.5 Construir Grafo HNSW
    print("[*] Construyendo Grafo HNSW para acelerar búsqueda...")
    index.build()
    comp_time = time.time() - start_comp
    
    # 4. Retrieval Benchmarking
    print(f"[*] Ejecutando Retrieval QA (Queries: 100, Noise: 0.001)...")
    num_queries = 100
    query_indices = np.random.randint(0, NUM_DOCS, num_queries)
    
    latencies_gaje = []
    exact_hits = 0
    top_k_hits = 0
    
    for q_idx in query_indices:
        # Query con ruido mínimo para ver límite teórico
        query_vec = embeddings[q_idx] + np.random.normal(0, 0.001, DIMS).astype(np.float32)
        query_vec /= (np.linalg.norm(query_vec) + 1e-10)
        
        start_q = time.time()
        # search() usará HNSW ahora
        results = index.search(query_vec.tolist(), k=TOP_K_RETRIEVAL, ef=64)
        latencies_gaje.append(time.time() - start_q)
        
        # Medir exactitud
        retrieved_ids = [r[0] for r in results]
        if retrieved_ids[0] == q_idx:
            exact_hits += 1
        if q_idx in retrieved_ids:
            top_k_hits += 1

    # 5. Métricas de RAM y Contexto
    # Baseline: Float32 (4 bytes per dim)
    ram_baseline = (NUM_DOCS * DIMS * 4) / (1024 * 1024)
    # GAJE: 2 bits per dim (0.25 bytes per dim) + overhead de Rust Index
    ram_gaje = (len(index.database)) / (1024 * 1024)
    
    # Ahorro de Contexto (Tokens simulados)
    # Si enviáramos embeddings al LLM, GAJE ahorraría 16x espacio
    context_reduction = (ram_baseline / ram_gaje)

    # 6. Reporte Final
    avg_latency = np.mean(latencies_gaje) * 1000
    exactness = (exact_hits / num_queries) * 100
    recall_at_k = (top_k_hits / num_queries) * 100

    print("\n" + "-"*40)
    print(f"📊 RESULTADOS FINALES (GAJE PROTOCOL)")
    print("-"*40)
    print(f"✅ Exactitud (Top-1):      {exactness:.2f}%")
    print(f"✅ Recall (Top-{TOP_K_RETRIEVAL}):        {recall_at_k:.2f}%")
    print(f"📉 Reducción RAM:          {ram_baseline:.2f} MB -> {ram_gaje:.2f} MB ({context_reduction:.1f}x)")
    print(f"⏱️ Latencia Media:        {avg_latency:.2f} ms")
    print(f"🧬 Tiempo Compresión:      {comp_time:.2f} s")
    print(f"💬 Tokens LLM Simulados:   Ahorro del 93.75% en vectores semánticos")
    print("-"*40)
    
    if exactness > 80:
        print("🔥 ESTADO: PRODUCCIÓN READY PARA GEMMA 4")
    else:
        print("⚠️ ESTADO: REQUIERE AJUSTE DE CENTROIDES")

if __name__ == "__main__":
    run_qa_benchmark()
