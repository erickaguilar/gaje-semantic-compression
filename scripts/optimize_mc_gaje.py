import os
import sys
import numpy as np
import time
import argparse

# Asegurar que se usa el paquete local
sys.path.append(os.path.abspath("python"))

from gaje.nn.stabilized import GenomicLLM
from gaje.nn.configs import get_config
def simulate_gaje_quantization(weights_f32, centroids):
    distances = np.abs(weights_f32[:, np.newaxis] - centroids)
    nearest_idx = np.argmin(distances, axis=1)
    reconstructed = centroids[nearest_idx]
    return np.mean((weights_f32 - reconstructed) ** 2)

def optimize_layer_mc(llm, layer, layer_name, iterations=2000, noise_scale=0.1):
    """
    Optimiza los centroides de una GenomicLayer usando Monte Carlo.
    """
    # En un caso real, optimizaríamos basándonos en la matriz original F32.
    # Dado que estamos en un modelo Born-Genomic o ya cuantizado, usaremos
    # una fila de muestra de la capa como nuestra "distribución representativa".
    
    # Extraemos una fila simulada para evaluar el MSE (idealmente usaríamos get_row de varias filas)
    try:
        sample_weights = layer.get_row(0)
    except:
        sample_weights = np.random.normal(0, 0.02, layer.in_features).astype(np.float32)
        
    std = np.std(sample_weights)
    if std == 0: std = 0.02
    
    best_centroids = np.array(layer.linear.centroids[:4])
    best_mse = simulate_gaje_quantization(sample_weights, best_centroids)
    
    improved = False
    
    for _ in range(iterations):
        mutation = np.random.normal(0, noise_scale * std, 4)
        candidate_centroids = best_centroids + mutation
        candidate_centroids.sort() # Mantener orden
        
        mse = simulate_gaje_quantization(sample_weights, candidate_centroids)
        if mse < best_mse:
            best_mse = mse
            best_centroids = candidate_centroids
            improved = True
            
    if improved:
        # Replicamos los 4 centroides a lo largo de todos los bloques de la matriz completa
        n_blocks_total = (layer.in_features * layer.out_features) // layer.block_size
        new_centroids = np.tile(best_centroids, n_blocks_total).astype(np.float32)
        # Forzamos la actualización en el objeto nativo a través del contenedor central RustGenomicLLM
        delta = new_centroids - np.array(layer.linear.centroids)
        llm.rust_llm.apply_mutation(layer_name, delta.tolist(), False)
        
    return improved, best_mse

def main():
    parser = argparse.ArgumentParser(description="🎲 GAJE Monte Carlo Optimizer")
    parser.add_argument("--arch", type=str, default="qwen2", help="Arquitectura a optimizar")
    parser.add_argument("--blocks", type=int, default=1, help="Número de bloques")
    parser.add_argument("--iterations", type=int, default=300, help="Número de generaciones de Monte Carlo")
    parser.add_argument("--out", type=str, default="models/mc_optimized_qwen", help="Directorio de salida")
    args = parser.parse_args()
    
    print(f"🎲 Iniciando Optimización Global Monte Carlo para {args.arch} ({args.iterations} generaciones)")
    config = get_config(args.arch)
    llm = GenomicLLM(model_path=None, num_blocks=args.blocks, config=config)
    
    print("[*] Organismo inicializado (Born-Genomic). Comenzando optimización de centroides...")
    start_time = time.time()
    
    layers_optimized = 0
    
    # 1. Optimizar Embeddings
    print(f"    -> Optimizando Token Embeddings...")
    improved, mse = optimize_layer_mc(llm, llm.embeddings, "token_embd", iterations=args.iterations)
    if improved: layers_optimized += 1
        
    # 2. Optimizar Bloques (Solo capas Atencionales por ejemplo)
    for i, block in enumerate(llm.blocks):
        print(f"    -> Optimizando Bloque {i} (Atención y FFN)...")
        p = f"blk.{i}."
        # Atención
        imp, _ = optimize_layer_mc(llm, block.attn_layer.q_gen, p+"attn_q", iterations=args.iterations); layers_optimized += int(imp)
        imp, _ = optimize_layer_mc(llm, block.attn_layer.k_gen, p+"attn_k", iterations=args.iterations); layers_optimized += int(imp)
        imp, _ = optimize_layer_mc(llm, block.attn_layer.v_gen, p+"attn_v", iterations=args.iterations); layers_optimized += int(imp)
        imp, _ = optimize_layer_mc(llm, block.attn_layer.w_o, p+"attn_output", iterations=args.iterations); layers_optimized += int(imp)
        # FFN
        imp, _ = optimize_layer_mc(llm, block.gate_gen, p+"ffn_gate", iterations=args.iterations); layers_optimized += int(imp)
        imp, _ = optimize_layer_mc(llm, block.up_gen, p+"ffn_up", iterations=args.iterations); layers_optimized += int(imp)
        imp, _ = optimize_layer_mc(llm, block.w_down, p+"ffn_down", iterations=args.iterations); layers_optimized += int(imp)
        
    # 3. Optimizar LM Head
    print("    -> Optimizando LM Head...")
    improved, mse = optimize_layer_mc(llm, llm.lm_head, "lm_head", iterations=args.iterations)
    if improved: layers_optimized += 1
        
    duration = time.time() - start_time
    print(f"✅ Optimización MC completada en {duration:.2f}s. Capas mejoradas: {layers_optimized}")
    
    # GUARDAR EL MODELO
    os.makedirs(args.out, exist_ok=True)
    print(f"📦 Guardando organismo optimizado en formato .gaje en: {args.out}")
    llm.save(args.out)
    print("¡Guardado exitoso!")

if __name__ == "__main__":
    main()
