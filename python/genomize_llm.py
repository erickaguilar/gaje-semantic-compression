import gguf
import numpy as np
import dna_semantic_compression
import time
import os
from transformers import AutoTokenizer

def dequantize_q8_0(data_u8, out_features, in_features):
    """
    De-cuantiza bloques Q8_0 (Block size 32) de GGUF a float32.
    """
    n_blocks = in_features // 32
    weights_f32 = np.zeros((out_features, in_features), dtype=np.float32)
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
    Capa de LLM con pesos genomizados a 2 bits (Soporta Block-Quant).
    """
    def __init__(self, name, tensor):
        self.in_features = tensor.shape[0]
        self.out_features = tensor.shape[1]
        
        print(f"[*] Genomizando capa '{name}' (In:{self.in_features} -> Out:{self.out_features})...")
        weights_f32 = dequantize_q8_0(tensor.data, self.out_features, self.in_features)
        
        # 1. Block-Quant: Calcular centroides por cada neurona (fila)
        all_centroids = []
        dna_batch = []
        
        start_q = time.time()
        for i in range(self.out_features):
            w = weights_f32[i]
            std = np.std(w)
            mean = np.mean(w)
            
            # Max-Lloyd 2-bit per neuron
            thresholds = [mean - 0.9816 * std, mean, mean + 0.9816 * std]
            centroids = [mean - 1.510 * std, mean - 0.4528 * std, mean + 0.4528 * std, mean + 1.510 * std]
            
            all_centroids.extend(centroids)
            dna_batch.append(dna_semantic_compression.quantize_embedding(w.tolist(), thresholds))
        
        self.engine = dna_semantic_compression.GajeIndex([], all_centroids)
        self.engine.add_batch(dna_batch)
        
        self.comp_time = time.time() - start_q
        print(f"    [+] Block-Quant (16x) completada en {self.comp_time:.2f}s")

    def forward(self, x):
        if isinstance(x, np.ndarray): x = x.tolist()
        return self.engine.genomic_linear_forward(x)

class GenomicAttentionLayer:
    """
    Capa de Atención Multi-Head acelerada en Rust (Soporta GQA + Block-Quant).
    """
    def __init__(self, reader, prefix):
        tensor_q = next(t for t in reader.tensors if t.name == prefix + "attn_q.weight")
        tensor_k = next(t for t in reader.tensors if t.name == prefix + "attn_k.weight")
        
        head_dim = 64
        self.n_heads_q = tensor_q.shape[1] // head_dim
        self.n_heads_kv = tensor_k.shape[1] // head_dim
        
        print(f"[*] GQA Config: Q_Heads={self.n_heads_q}, KV_Heads={self.n_heads_kv}, Head_Dim={head_dim}")

        def get_dna_and_centroids(name):
            tensor = next(t for t in reader.tensors if t.name == prefix + name + ".weight")
            w_f32 = dequantize_q8_0(tensor.data, tensor.shape[1], tensor.shape[0])
            
            packed_rows = []
            layer_centroids = []
            
            for row in w_f32:
                std = np.std(row)
                mean = np.mean(row)
                thresholds = [mean - 0.9816 * std, mean, mean + 0.9816 * std]
                centroids = [mean - 1.510 * std, mean - 0.4528 * std, mean + 0.4528 * std, mean + 1.510 * std]
                
                layer_centroids.extend(centroids)
                packed_rows.append(dna_semantic_compression.quantize_embedding(row.tolist(), thresholds))
                
            return b"".join(packed_rows), layer_centroids, tensor.shape[0] // 4

        print(f"[*] Genomizando proyecciones Q, K, V para {prefix} (Block-Quant)...")
        w_q_dna, c_q, stride = get_dna_and_centroids("attn_q")
        w_k_dna, c_k, _ = get_dna_and_centroids("attn_k")
        w_v_dna, c_v, _ = get_dna_and_centroids("attn_v")
        
        all_centroids = c_q + c_k + c_v
        
        self.kernel = dna_semantic_compression.GenomicAttention(
            w_q_dna, w_k_dna, w_v_dna, all_centroids, stride, self.n_heads_q, self.n_heads_kv
        )

    def forward(self, x, pos):
        return self.kernel.forward(x, pos)

class GenomicTransformerBlock:
    def __init__(self, block_idx, reader):
        self.block_idx = block_idx
        self.layers = {}
        prefix = f"blk.{block_idx}."
        print(f"\n🧬 [Bloque {block_idx}] Genomizando con Block-Quant...")
        
        # 1. Inicializar Atención Acelerada
        self.attn = GenomicAttentionLayer(reader, prefix)
        
        # 2. Inicializar FFN
        for name in ["ffn_up", "ffn_down"]:
            tensor = next(t for t in reader.tensors if t.name == prefix + name + ".weight")
            self.layers[name] = GenomicLLMLayer(tensor.name, tensor)

    def rms_norm(self, x, eps=1e-6):
        x = np.array(x)
        rms = np.sqrt(np.mean(x**2) + eps)
        return (x / rms).tolist()

    def forward(self, x, pos):
        x_in = np.array(x)
        attn_out = np.array(self.attn.forward(x_in.tolist(), pos))
        x_mid = self.rms_norm(x_in + attn_out)
        
        ffn_up = np.array(self.layers['ffn_up'].forward(x_mid))
        ffn_up = np.maximum(0, ffn_up) # ReLU
        ffn_down = np.array(self.layers['ffn_down'].forward(ffn_up.tolist()))
        
        x_final = self.rms_norm(np.array(x_mid) + ffn_down[:len(x_mid)])
        return x_final

class GenomicLLM:
    def __init__(self, model_path, num_blocks=1):
        self.reader = gguf.GGUFReader(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
        
        print("\n[*] Preparando matriz de Embeddings...")
        embd_tensor = next(t for t in self.reader.tensors if t.name == "token_embd.weight")
        self.embedding_matrix = dequantize_q8_0(embd_tensor.data, embd_tensor.shape[1], embd_tensor.shape[0])
        
        self.blocks = [GenomicTransformerBlock(i, self.reader) for i in range(num_blocks)]

    def generate(self, prompt, max_new_tokens=20):
        input_ids = self.tokenizer.encode(prompt)
        print(f"\n🚀 Generando con Block-Quant: '{prompt}'")
        
        generated_ids = []
        for i in range(max_new_tokens):
            last_id = input_ids[-1]
            x = self.embedding_matrix[last_id].tolist()
            
            for j, block in enumerate(self.blocks):
                x = block.forward(x, len(input_ids) - 1)
            
            logits = np.dot(self.embedding_matrix, x)
            next_id = np.argmax(logits)
            
            if next_id == self.tokenizer.eos_token_id: break
            
            generated_ids.append(next_id)
            input_ids.append(next_id)
            
            word = self.tokenizer.decode([next_id])
            print(word, end="", flush=True)
            
        return self.tokenizer.decode(generated_ids)

if __name__ == "__main__":
    model_path = "/data/data/com.termux/files/home/models/qwen2-0_5b-q8_0.gguf"
    model = GenomicLLM(model_path, num_blocks=2)
    model.generate("El protocolo GAJE es", max_new_tokens=10)
