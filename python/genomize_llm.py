import gguf
import numpy as np
import dna_semantic_compression
import time
import os

def dequantize_q8_0(data_u8, out_features, in_features):
    """
    De-cuantiza bloques Q8_0 (Block size 32) de GGUF a float32.
    Cada bloque: 2 bytes (float16 delta) + 32 bytes (int8 weights).
    """
    n_blocks = in_features // 32
    weights_f32 = np.zeros((out_features, in_features), dtype=np.float32)
    
    # El data_u8 suele venir como un array plano de bytes
    data_raw = data_u8.view(np.uint8).reshape(out_features, -1)
    
    for i in range(out_features):
        row_data = data_raw[i]
        for b in range(n_blocks):
            offset = b * 34
            delta = np.frombuffer(row_data[offset:offset+2], dtype=np.float16)[0].astype(np.float32)
            qs = row_data[offset+2:offset+34].view(np.int8).astype(np.float32)
            weights_f32[i, b*32 : (b+1)*32] = qs * delta
    return weights_f32

class GenomicLLMLayer:
    """
    Capa de LLM (ej. Atención o FFN) con pesos genomizados a 2 bits.
    """
    def __init__(self, name, tensor):
        # GGUF: shape[0]=in, shape[1]=out
        self.in_features = tensor.shape[0]
        self.out_features = tensor.shape[1]
        
        print(f"[*] Genomizando capa '{name}' (In:{self.in_features} -> Out:{self.out_features})...")
        
        # 1. De-cuantizar Q8_0 a F32
        weights_f32 = dequantize_q8_0(tensor.data, self.out_features, self.in_features)
        
        # 2. Entrenamiento y Cuantización a 2-bit DNA
        flat_weights = weights_f32.flatten()
        std = np.std(flat_weights)
        mean = np.mean(flat_weights)
        
        self.thresholds = [mean - 0.9816 * std, mean, mean + 0.9816 * std]
        self.centroids = [mean - 1.510 * std, mean - 0.4528 * std, mean + 0.4528 * std, mean + 1.510 * std]
        
        # 3. Convertir a ADN
        start_q = time.time()
        dna_batch = [
            dna_semantic_compression.quantize_embedding(w.tolist(), self.thresholds)
            for w in weights_f32
        ]
        
        self.engine = dna_semantic_compression.GajeIndex([], self.centroids)
        self.engine.add_batch(dna_batch)
        
        self.comp_time = time.time() - start_q
        print(f"    [+] Compresión 16x completada en {self.comp_time:.2f}s")

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = x.tolist()
        return self.engine.genomic_linear_forward(x)

class GenomicAttentionLayer:
    """
    Capa de Atención Multi-Head acelerada en Rust (Soporta GQA).
    """
    def __init__(self, reader, prefix, centroids):
        # 1. Extraer dimensiones y cabezas
        tensor_q = next(t for t in reader.tensors if t.name == prefix + "attn_q.weight")
        tensor_k = next(t for t in reader.tensors if t.name == prefix + "attn_k.weight")
        
        in_features = tensor_q.shape[0]
        head_dim = 64
        self.n_heads_q = tensor_q.shape[1] // head_dim
        self.n_heads_kv = tensor_k.shape[1] // head_dim
        
        print(f"[*] GQA Config: Q_Heads={self.n_heads_q}, KV_Heads={self.n_heads_kv}, Head_Dim={head_dim}")

        # 2. Extraer y empaquetar pesos Q, K, V
        def get_dna_weights(name):
            tensor = next(t for t in reader.tensors if t.name == prefix + name + ".weight")
            out_f = tensor.shape[1]
            in_f = tensor.shape[0]
            
            w_f32 = dequantize_q8_0(tensor.data, out_f, in_f)
            
            std = np.std(w_f32)
            mean = np.mean(w_f32)
            thresholds = [mean - 0.9816 * std, mean, mean + 0.9816 * std]
            
            packed_rows = [
                dna_semantic_compression.quantize_embedding(row.tolist(), thresholds)
                for row in w_f32
            ]
            return b"".join(packed_rows), in_f // 4

        print(f"[*] Genomizando proyecciones Q, K, V para {prefix}...")
        w_q_dna, stride = get_dna_weights("attn_q")
        w_k_dna, _ = get_dna_weights("attn_k")
        w_v_dna, _ = get_dna_weights("attn_v")
        
        self.kernel = dna_semantic_compression.GenomicAttention(
            w_q_dna, w_k_dna, w_v_dna, centroids, stride, self.n_heads_q, self.n_heads_kv
        )

    def forward(self, x):
        return self.kernel.forward(x)

class GenomicTransformerBlock:
    """
    Representa un bloque completo de Transformer (Attn + FFN) genomizado.
    """
    def __init__(self, block_idx, reader):
        self.block_idx = block_idx
        self.layers = {}
        prefix = f"blk.{block_idx}."
        
        print(f"\n🧬 [Bloque {block_idx}] Iniciando Genomización Integral...")
        self.centroids = [-1.510, -0.4528, 0.4528, 1.510]
        
        # 1. Inicializar Atención Acelerada
        self.attn = GenomicAttentionLayer(reader, prefix, self.centroids)
        
        # 2. Inicializar FFN
        for name in ["ffn_up", "ffn_down"]:
            tensor = next(t for t in reader.tensors if t.name == prefix + name + ".weight")
            self.layers[name] = GenomicLLMLayer(tensor.name, tensor)

    def rms_norm(self, x, eps=1e-6):
        x = np.array(x)
        rms = np.sqrt(np.mean(x**2) + eps)
        return (x / rms).tolist()

    def forward(self, x):
        x_in = np.array(x)
        attn_out = np.array(self.attn.forward(x_in.tolist()))
        x_mid = self.rms_norm(x_in + attn_out)
        
        ffn_up = np.array(self.layers['ffn_up'].forward(x_mid))
        ffn_up = np.maximum(0, ffn_up) # ReLU
        ffn_down = np.array(self.layers['ffn_down'].forward(ffn_up.tolist()))
        
        x_final = self.rms_norm(np.array(x_mid) + ffn_down[:len(x_mid)])
        return x_final

def run_deep_propagation_test(model_path, num_blocks=3):
    if not os.path.exists(model_path):
        print(f"❌ Error: {model_path} no encontrado.")
        return

    reader = gguf.GGUFReader(model_path)
    blocks = []
    for i in range(num_blocks):
        blocks.append(GenomicTransformerBlock(i, reader))
        
    in_dim = blocks[0].attn.n_heads_q * 64 # head_dim
    print(f"\n🚀 INICIANDO TEST DE PROPAGACIÓN PROFUNDA ({num_blocks} Bloques)")
    print(f"[*] Dimensión de Señal: {in_dim}")
    
    x_gen = np.random.normal(0, 0.1, (in_dim,)).astype(np.float32)
    
    print("\n[*] Propagando señal a través de la cadena genómica...")
    for i, block in enumerate(blocks):
        start = time.perf_counter()
        x_gen = block.forward(x_gen)
        end = time.perf_counter()
        signal_power = np.linalg.norm(x_gen)
        print(f"    [Bloque {i}] Inferencia: {(end-start)*1000:.2f} ms | Energía Señal: {signal_power:.4f}")

    print(f"\n✅ Propagación de {num_blocks} bloques completada.")

if __name__ == "__main__":
    model_path = "/data/data/com.termux/files/home/models/qwen2-0_5b-q8_0.gguf"
    run_deep_propagation_test(model_path, num_blocks=3)
