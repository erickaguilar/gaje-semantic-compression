from gaje.core import _impl as dna_semantic_compression
import numpy as np
import time

class GenomicLayer:
    def __init__(self, in_features, out_features, weights=None):
        self.in_features = in_features
        self.out_features = out_features
        
        # 1. Inicializar o recibir pesos
        if weights is None:
            weights = np.random.normal(0, 0.02, (out_features, in_features)).astype(np.float32)
        
        # 2. Entrenar Código Genético (Adaptativo)
        print(f"[*] Entrenando Capa Genómica ({in_features}x{out_features})...")
        self.thresholds, self.centroids = self._train(weights)
        
        # 3. Cuantizar y Cargar en Rust Index
        dna_weights = [
            dna_semantic_compression.quantize_embedding(w.tolist(), self.thresholds)
            for w in weights
        ]
        self.index = dna_semantic_compression.GajeIndex([], self.centroids)
        self.index.add_batch(dna_weights)
        
    def _train(self, weights):
        all_t, all_c = [], []
        for d in range(weights.shape[1]):
            col = weights[:, d]
            t = np.percentile(col, [25, 50, 75]).tolist()
            c = [
                np.mean(col[col < t[0]]),
                np.mean(col[(col >= t[0]) & (col < t[1])]),
                np.mean(col[(col >= t[1]) & (col < t[2])]),
                np.mean(col[col >= t[2]])
            ]
            all_t.extend(t)
            all_c.extend([float(x) if not np.isnan(x) else 0.0 for x in c])
        return all_t, all_c

    def forward(self, x):
        # x es un vector de Python (lista)
        return self.index.genomic_linear_forward(x)

class GenomicMLP:
    def __init__(self, layers_config):
        """
        layers_config: lista de tamaños [input, hidden1, hidden2, ..., output]
        """
        print(f"🧬 Construyendo MLP Genómico: {layers_config}")
        self.layers = []
        for i in range(len(layers_config) - 1):
            self.layers.append(GenomicLayer(layers_config[i], layers_config[i+1]))
            
    def forward(self, x):
        curr_x = x.tolist() if isinstance(x, np.ndarray) else x
        for i, layer in enumerate(self.layers):
            # Forward Genómico
            curr_x = layer.forward(curr_x)
            
            # Activación ReLU (excepto en la última capa)
            if i < len(self.layers) - 1:
                curr_x = [max(0.0, val) for val in curr_x]
        return curr_x

def run_mlp_demo():
    print("\n" + "="*60)
    print("🚀 GAJE PROTOCOL: NATIVE GENOMIC MLP INFERENCE")
    print("="*60)
    
    # Configuración: 768 (Input) -> 512 (Hidden) -> 128 (Hidden) -> 10 (Output/Clases)
    config = [768, 512, 128, 10]
    mlp = GenomicMLP(config)
    
    # Simular una entrada (ej. un embedding de texto)
    x = np.random.normal(0, 1.0, (768,)).astype(np.float32)
    
    print("\n[*] Ejecutando Inferencia en Cascada (3 capas genómicas)...")
    start_time = time.perf_counter()
    output = mlp.forward(x)
    duration = (time.perf_counter() - start_time) * 1000
    
    print(f"\n✅ Inferencia completada en {duration:.2f} ms")
    print(f"✅ Resultado (Primeras 5 clases): {output[:5]}")
    
    # Calcular ahorro de RAM
    total_weights = sum([l.in_features * l.out_features for l in mlp.layers])
    ram_f32 = (total_weights * 4) / (1024 * 1024)
    ram_gen = (total_weights * 0.25) / (1024 * 1024) # 2 bits = 0.25 bytes
    
    print(f"\n📉 RAM del Modelo (Float32): {ram_f32:.2f} MB")
    print(f"📉 RAM del Modelo (Genomic): {ram_gen:.2f} MB")
    print(f"🔥 Factor de Compresión:     16x")
    print("="*60)

if __name__ == "__main__":
    run_mlp_demo()
