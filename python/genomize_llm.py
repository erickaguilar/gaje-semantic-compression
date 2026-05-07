import gguf
import numpy as np
import dna_semantic_compression
import time
import os

class GenomicLLMLayer:
    """
    Capa de LLM (ej. Atención o FFN) con pesos genomizados a 2 bits.
    """
    def __init__(self, name, weights_f32, group_size=32):
        self.name = name
        self.out_features, self.in_features = weights_f32.shape
        
        print(f"[*] Genomizando capa '{name}' ({self.in_features}x{self.out_features})...")
        
        # 1. Entrenamiento y Cuantización (Max-Lloyd 1D)
        # Por simplicidad en este prototipo, usamos centroides globales
        # En una versión final, usaríamos Block-Quant
        flat_weights = weights_f32.flatten()
        std = np.std(flat_weights)
        mean = np.mean(flat_weights)
        
        # Max-Lloyd optimizado para Normal
        self.thresholds = [mean - 0.9816 * std, mean, mean + 0.9816 * std]
        self.centroids = [mean - 1.510 * std, mean - 0.4528 * std, mean + 0.4528 * std, mean + 1.510 * std]
        
        # 2. Convertir a ADN y cargar en motor Rust
        start_q = time.time()
        dna_batch = [
            dna_semantic_compression.quantize_embedding(w.tolist(), self.thresholds)
            for w in weights_f32
        ]
        
        # Usamos GajeIndex como contenedor de pesos de 2 bits
        self.engine = dna_semantic_compression.GajeIndex([], self.centroids)
        self.engine.add_batch(dna_batch)
        
        self.comp_time = time.time() - start_q
        print(f"    [+] Compresión 16x completada en {self.comp_time:.2f}s")

    def forward(self, x):
        """
        Inferencia ultra-ligera usando LUT-ADC en Rust.
        """
        # Asegurar que x sea una lista para el motor Rust
        if isinstance(x, np.ndarray):
            x = x.tolist()
        return self.engine.genomic_linear_forward(x)

class GenomicTransformerBlock:
    """
    Representa un bloque completo de Transformer (Attn + FFN) genomizado.
    """
    def __init__(self, block_idx, reader):
        self.block_idx = block_idx
        self.layers = {}
        
        # Prefijo de las capas para este bloque en Qwen2/Llama
        prefix = f"blk.{block_idx}."
        
        print(f"\n🧬 [Bloque {block_idx}] Iniciando Genomización Integral...")
        
        for tensor in reader.tensors:
            if tensor.name.startswith(prefix) and "weight" in tensor.name and len(tensor.shape) == 2:
                # Extraer nombre corto (ej. attn_q, ffn_down)
                short_name = tensor.name.replace(prefix, "").replace(".weight", "")
                weights_f32 = tensor.data.astype(np.float32)
                
                # Genomizar capa
                self.layers[short_name] = GenomicLLMLayer(tensor.name, weights_f32)

    def rms_norm(self, x, eps=1e-6):
        """
        Normalización Genómica (Root Mean Square Norm).
        Mantiene la señal estable entre bloques.
        """
        x = np.array(x)
        # Calcular RMS
        rms = np.sqrt(np.mean(x**2) + eps)
        # Normalizar
        return (x / rms).tolist()

    def forward(self, x):
        """
        Inferencia estabilizada con RMSNorm y Residual Connections.
        """
        x_in = np.array(x)
        
        # 1. Rama de Atención (Simplificada: q*x)
        attn_out = np.array(self.layers['attn_q'].forward(x_in.tolist()))
        attn_impact = attn_out[:len(x_in)]
        
        # Residual + Normalización
        x_mid = self.rms_norm(x_in + attn_impact)
        
        # 2. Rama FFN (ffn_up -> ffn_down)
        ffn_up = np.array(self.layers['ffn_up'].forward(x_mid))
        ffn_up = np.maximum(0, ffn_up) # ReLU
        ffn_down = np.array(self.layers['ffn_down'].forward(ffn_up.tolist()))
        
        # Final Residual + Final Normalización
        x_final = self.rms_norm(np.array(x_mid) + ffn_down[:len(x_mid)])
        
        return x_final

def run_deep_propagation_test(model_path, num_blocks=3):
    if not os.path.exists(model_path):
        print(f"❌ Error: {model_path} no encontrado.")
        return

    reader = gguf.GGUFReader(model_path)
    blocks = []
    
    # 1. Genomizar secuencia de bloques
    for i in range(num_blocks):
        blocks.append(GenomicTransformerBlock(i, reader))
        
    # 2. Referencia Float32 (Capa 0 para dimensionamiento)
    in_dim = list(blocks[0].layers.values())[0].in_features
    h_dim = list(blocks[0].layers.values())[0].out_features # FFN Up suele ser la más ancha
    
    print(f"\n🚀 INICIANDO TEST DE PROPAGACIÓN PROFUNDA ({num_blocks} Bloques)")
    print(f"[*] Dimensión de Señal: {in_dim}")
    
    # Señal inicial (Embedding simulado)
    x_gen = np.random.normal(0, 0.1, (896,)).astype(np.float32) # Dimensión típica de Qwen2-0.5B
    x_orig = x_gen.copy()
    
    # 3. Propagación a través de la cadena de ADN
    print("\n[*] Propagando señal a través de la cadena genómica...")
    
    for i, block in enumerate(blocks):
        start = time.perf_counter()
        x_gen = block.forward(x_gen)
        end = time.perf_counter()
        
        # Medir "salud" de la señal tras cada bloque
        # (En un test real compararíamos contra x_orig transformado por F32)
        # Aquí medimos la energía de la señal
        signal_power = np.linalg.norm(x_gen)
        print(f"    [Bloque {i}] Inferencia: {(end-start)*1000:.2f} ms | Energía Señal: {signal_power:.4f}")

    print(f"\n✅ Propagación de {num_blocks} bloques completada.")
    print(f"--- Resumen de Recursos ---")
    total_ram = sum([sum([len(ly.engine.database) for ly in bl.layers.values()]) for bl in blocks])
    print(f"🔥 RAM Genómica Total ({num_blocks} bloques): {total_ram/(1024*1024):.2f} MB")
    print(f"📉 RAM F32 Estimada: {(total_ram/(1024*1024))*16:.2f} MB")

if __name__ == "__main__":
    model_path = "/data/data/com.termux/files/home/models/qwen2-0_5b-q8_0.gguf"
    run_deep_propagation_test(model_path, num_blocks=3)
