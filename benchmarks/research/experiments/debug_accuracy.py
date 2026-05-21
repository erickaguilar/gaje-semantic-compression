from gaje.core import _impl as dna_semantic_compression
import numpy as np

def debug_accuracy():
    print("🧪 DEBUG: GAJE FIDELITY TEST (No Noise)")
    dims = 128
    n = 100
    
    # 1. Datos aleatorios pero fijos
    np.random.seed(42)
    data = np.random.normal(0, 1, (n, dims)).astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    
    # 2. Entrenamiento Per-Dimension
    all_thresholds = []
    all_centroids = []
    for d in range(dims):
        col = data[:, d]
        t = np.percentile(col, [25, 50, 75])
        all_thresholds.extend(t.tolist())
        all_centroids.extend([
            float(np.mean(col[col < t[0]])),
            float(np.mean(col[(col >= t[0]) & (col < t[1])])),
            float(np.mean(col[(col >= t[1]) & (col < t[2])])),
            float(np.mean(col[col >= t[2]]))
        ])
    
    # 3. Indexación
    print(f"[*] Indexando {n} vectores...")
    index = dna_semantic_compression.GajeIndex([], all_centroids)
    db_dna = [dna_semantic_compression.quantize_embedding(v.tolist(), all_thresholds) for v in data]
    index.add_batch(db_dna)
    
    # 4. Búsqueda de identidad (Query = Vector original)
    print("[*] Validando recuperación de identidad...")
    hits = 0
    for i in range(n):
        query = data[i].tolist()
        results = index.search(query, k=1)
        if results[0][0] == i:
            hits += 1
            
    accuracy = (hits / n) * 100
    print(f"🎯 Exactitud de Identidad: {accuracy:.2f}%")
    
    if accuracy < 90:
        print("❌ ERROR DETECTADO: El motor de búsqueda no recupera identidades.")
    else:
        print("✅ EL MOTOR ES CORRECTO: El problema es el ruido o la escala.")

if __name__ == "__main__":
    debug_accuracy()
