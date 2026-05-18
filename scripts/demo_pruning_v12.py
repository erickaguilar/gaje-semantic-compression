import os
import sys
import numpy as np
import time

# Asegurar que se usa el paquete local
sys.path.append(os.path.abspath("python"))

from gaje.processing.balancer import SignalToNoiseBalancer
from gaje.core import _impl as dna_core

def calculate_shannon_entropy(data):
    """
    Simula el cálculo de entropía de Shannon por dimensión.
    En un caso real, esto analizaría la varianza de los pesos originales.
    """
    # Para el demo, generamos entropía aleatoria pero con algunas dimensiones "muertas" (casi 0)
    entropy = np.random.uniform(0.1, 1.0, data.shape[1])
    # Forzamos un 30% de dimensiones muertas (entropía muy baja)
    dead_mask = np.random.choice([0, 1], size=data.shape[1], p=[0.3, 0.7])
    entropy *= dead_mask
    return entropy

def main():
    print("🧬 GAJE PHASE 12: NEURAL PRUNING DNA TEST")
    print("="*60)
    
    # 1. Crear una matriz de pesos simulada (ej: 1024 x 1024)
    rows, cols = 1024, 1024
    print(f"[*] Matriz Original: {rows}x{cols} (Float32)")
    weights = np.random.normal(0, 0.02, (rows, cols)).astype(np.float32)
    original_size = weights.nbytes / (1024*1024)
    print(f"    Tamaño en RAM: {original_size:.2f} MB")
    
    # 2. Genomización inicial (2-bit puro)
    print(f"\n[*] Genomizando a 2 bits (DNA)...")
    dna_strands = []
    thresholds = [-0.68, 0.0, 0.68] # Centroides base
    for row in weights:
        dna = dna_core.quantize_embedding(row.tolist(), thresholds)
        dna_strands.append(dna)
    
    # Flatten database
    database = b"".join(dna_strands)
    stride = len(dna_strands[0])
    genomic_size = len(database) / (1024*1024)
    print(f"    Tamaño Genómico (2-bit): {genomic_size:.2f} MB (Reducción {(1 - genomic_size/original_size)*100:.1f}%)")
    
    # 3. Análisis de Entropía (Fase 12)
    print(f"\n[*] Ejecutando Entropy Analyzer...")
    entropy_per_dim = calculate_shannon_entropy(weights)
    
    # 4. Neural Pruning DNA
    balancer = SignalToNoiseBalancer()
    print(f"[*] Aplicando Poda Genómica (Neural Pruning)...")
    start_prune = time.time()
    
    # El balancer usa el core nativo de Rust para repaquetar los bits
    pruned_db, active_dims = balancer.prune_dimensions(database, stride, entropy_per_dim, threshold=0.05)
    
    end_prune = time.time()
    pruned_size = len(pruned_db) / (1024*1024)
    
    print(f"\n[+] Resultados de la Fase 12:")
    print(f"    Dimensiones Originales: {cols}")
    print(f"    Dimensiones Activas:    {len(active_dims)}")
    print(f"    Dimensiones Eliminadas: {cols - len(active_dims)}")
    print(f"    Tamaño Final (Pruned):  {pruned_size:.2f} MB")
    print(f"    Ahorro Adicional:       {(1 - pruned_size/genomic_size)*100:.1f}%")
    print(f"    Tiempo de Ejecución:    {(end_prune - start_prune)*1000:.2f} ms")
    
    print("\n✅ TEST COMPLETADO: La poda genómica ha optimizado el organismo.")
    print("="*60)

if __name__ == "__main__":
    main()
