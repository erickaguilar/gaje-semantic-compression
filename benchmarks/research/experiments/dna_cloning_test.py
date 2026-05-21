import numpy as np
import os

def analyze_anchors():
    print("🧬 EXPERIMENTO: CLONACIÓN DE ANCLAS SEMÁNTICAS")
    print("-" * 50)
    
    # Intentamos cargar un bloque real
    path = "dna-semantic-compression/gaje_qwen2_full_v1/block_0/attn_w_q.bin"
    if not os.path.exists(path):
        # Fallback a datos sintéticos si el archivo no es legible directamente
        print("[!] Usando simulación de pesos Qwen2...")
        weights = np.random.normal(0, 0.1, (512, 512)).astype(np.float32)
    else:
        # Cargar datos binarios (asumimos float32 para la prueba de concepto)
        weights = np.fromfile(path, dtype=np.uint8).astype(np.float32) / 255.0
        # Normalizar para simular distribución de pesos
        weights = (weights - 0.5) * 2.0 

    # 1. LOCALIZACIÓN DE ANCLAS (La "Llave")
    # Las anclas son los pesos con mayor magnitud (los que más activan las neuronas)
    threshold = np.percentile(np.abs(weights), 95) # Top 5%
    anchors_mask = np.abs(weights) > threshold
    
    print(f"[*] Localizadas {np.sum(anchors_mask)} anclas críticas (Top 5%).")

    # 2. COMPRESIÓN ESTÁNDAR (2 bits / 1 base ADN)
    # Centroides típicos de GAJE: [-1.5, -0.5, 0.5, 1.5] normalizados
    centroids = np.array([-1.0, -0.33, 0.33, 1.0])
    
    def quantize(w, c):
        # Encuentra el centroide más cercano para cada peso
        idx = np.abs(w[:, None] - c).argmin(axis=-1)
        return c[idx]

    weights_flat = weights.flatten()
    reconstructed_std = quantize(weights_flat, centroids)
    
    mse_std = np.mean((weights_flat - reconstructed_std)**2)
    cos_std = np.dot(weights_flat, reconstructed_std) / (np.linalg.norm(weights_flat) * np.linalg.norm(reconstructed_std))

    # 3. CLONACIÓN CELULAR (Selective High Precision)
    # "Clonamos" las anclas con 4 bits (2 bases ADN) -> 16 centroides
    # El resto se queda en 2 bits.
    centroids_high = np.linspace(-1.0, 1.0, 16)
    
    reconstructed_cloned = reconstructed_std.copy()
    
    # Aplicamos "clonación" solo a las anclas
    anchors_flat = anchors_mask.flatten()
    reconstructed_cloned[anchors_flat] = quantize(weights_flat[anchors_flat], centroids_high)

    mse_cloned = np.mean((weights_flat - reconstructed_cloned)**2)
    cos_cloned = np.dot(weights_flat, reconstructed_cloned) / (np.linalg.norm(weights_flat) * np.linalg.norm(reconstructed_cloned))

    # 4. RESULTADOS
    print("\n" + "="*45)
    print(f"📊 RESULTADO DE LA CLONACIÓN")
    print("-" * 45)
    print(f"MÉTRICA         | ESTÁNDAR (2b) | CLONADO (Mix)")
    print(f"Error (MSE)     | {mse_std:.6f}      | {mse_cloned:.6f} {'🔥' if mse_cloned < mse_std else ''}")
    print(f"Similitud Cos   | {cos_std:.4f}        | {cos_cloned:.4f} {'🚀' if cos_cloned > cos_std else ''}")
    print("-" * 45)
    
    improvement = (cos_cloned - cos_std) / (1 - cos_std) * 100
    print(f"💡 GANANCIA DE FIDELIDAD: {improvement:.2f}%")
    print(f"📦 IMPACTO EN ESPAÑA: +0.1 bits/dim (Insigificante)")
    print("="*45)

if __name__ == "__main__":
    analyze_anchors()
