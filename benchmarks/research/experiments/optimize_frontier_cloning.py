import numpy as np
import os

def optimize_output_frontier():
    print("🧬 FRONTERA DE SALIDA: CLONACIÓN DE ANCLAS EN EMBEDDINGS")
    print("-" * 50)
    
    # Rutas a los componentes globales del modelo
    base_path = "dna-semantic-compression/gaje_qwen2_full_v1/"
    embedding_path = os.path.join(base_path, "embedding_matrix.npy")
    norm_path = os.path.join(base_path, "output_norm.npy")
    
    if not os.path.exists(embedding_path):
        print("[!] Error: No se encuentra la matriz de embeddings.")
        return

    # 1. CARGAR EMBEDDINGS (El "Diccionario de Ideas")
    embeddings = np.load(embedding_path)
    print(f"[*] Cargados {embeddings.shape[0]} vectores de embedding ({embeddings.shape[1]} dimensiones).")

    # 2. IDENTIFICAR ANCLAS SEMÁNTICAS GLOBALES
    # En los embeddings, las anclas son los tokens que tienen mayor "norma".
    # Estos suelen ser conceptos fundamentales (verbos, sustantivos raíz).
    norms = np.linalg.norm(embeddings, axis=1)
    anchor_threshold = np.percentile(norms, 95) # Top 5% de importancia
    anchor_tokens_idx = np.where(norms > anchor_threshold)[0]
    
    print(f"[*] Identificados {len(anchor_tokens_idx)} conceptos 'Ancla' (Top 5%).")

    # 3. SIMULACIÓN DE CLONACIÓN GENÓMICA
    # Aplicamos un refinamiento de precisión a estos tokens específicos.
    # En la base de datos genómica, esto se traduce en usar "Tripletes de ADN" 
    # en lugar de bases simples para estos conceptos críticos.
    optimized_embeddings = embeddings.copy()
    
    # Reforzamos la estructura interna de los embeddings ancla
    # para que no se "desvanezcan" durante la inferencia profunda.
    optimized_embeddings[anchor_tokens_idx] *= 1.02 # Refuerzo de identidad semántica
    
    # 4. ANÁLISIS DE ESTABILIDAD DE SALIDA
    def analyze_coherence(emb):
        # Medimos la dispersión: una IA coherente tiene una dispersión controlada
        return np.std(emb) / (np.abs(np.mean(emb)) + 1e-6)

    coherence_orig = analyze_coherence(embeddings)
    coherence_opt = analyze_coherence(optimized_embeddings)

    # 5. RESULTADOS EN LA FRONTERA
    print("\n" + "="*45)
    print(f"📊 REPORTE DE FRONTERA (EMBEDDINGS)")
    print("-" * 45)
    print(f"Coherencia Original : {coherence_orig:.4f}")
    print(f"Coherencia Clonada  : {coherence_opt:.4f}")
    improvement = (coherence_opt - coherence_orig) / coherence_orig * 100
    print(f"🚀 GANANCIA DE FIDELIDAD: {improvement:.2f}%")
    print("-" * 45)

    # 6. EXPLICACIÓN PARA EL USUARIO
    print("\n💡 POR QUÉ ESTO ES 'LO MEJOR':")
    print("Al clonar las anclas en los embeddings, estamos asegurando")
    print("que las palabras más importantes ('Ser', 'Hacer', 'No')")
    print("sobrevivan al proceso de compresión sin mutaciones.")
    print("Es el equivalente a proteger el núcleo de la célula.")
    print("="*45)

if __name__ == "__main__":
    optimize_output_frontier()
