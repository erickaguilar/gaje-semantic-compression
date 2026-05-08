import gguf
import numpy as np
import dna_semantic_compression
import time
import os
from transformers import AutoTokenizer

def dequantize_q8_0(data_u8, out_features, in_features):
    """
    De-cuantiza bloques Q8_0 usando el motor acelerado en Rust.
    """
    if isinstance(data_u8, np.ndarray):
        data_bytes = data_u8.tobytes()
    else:
        data_bytes = data_u8

    flat_weights = dna_semantic_compression.dequantize_q8_0_native(
        data_bytes, out_features, in_features
    )
    return np.array(flat_weights, dtype=np.float32).reshape(out_features, in_features)

class GenomicLLMLayer:
    """
    Capa de LLM con soporte para modo Genómico (2-bit) o Referencia (F32).
    """
    def __init__(self, name, tensor=None, database=None, centroids=None, mode='genomic'):
        self.mode = mode
        if tensor is not None:
            self.in_features = tensor.shape[0]
            self.out_features = tensor.shape[1]
            
            # Siempre cargamos pesos F32 para referencia o para genomizar
            self.weights_f32 = dequantize_q8_0(tensor.data, self.out_features, self.in_features)
            
            if mode == 'genomic':
                print(f"[*] Genomizando capa '{name}' (In:{self.in_features} -> Out:{self.out_features})...")
                all_centroids = []
                dna_batch = []
                for i in range(self.out_features):
                    w = self.weights_f32[i]
                    std = np.std(w); mean = np.mean(w)
                    thresholds = [mean - 0.9816 * std, mean, mean + 0.9816 * std]
                    centroids_row = [mean - 1.510 * std, mean - 0.4528 * std, mean + 0.4528 * std, mean + 1.510 * std]
                    all_centroids.extend(centroids_row)
                    dna_batch.append(dna_semantic_compression.quantize_embedding(w.tolist(), thresholds))
                
                self.engine = dna_semantic_compression.GajeIndex([], all_centroids)
                self.engine.add_batch(dna_batch)
        else:
            # Cargar desde datos persistidos (siempre genómico)
            self.engine = dna_semantic_compression.GajeIndex([], centroids.tolist())
            self.engine.add_batch([database])

    def forward(self, x):
        if self.mode == 'f32':
            return np.dot(self.weights_f32, x).tolist()
        
        if isinstance(x, np.ndarray): x = x.tolist()
        return self.engine.genomic_linear_forward(x)

class GenomicAttentionLayer:
    """
    Capa de Atención Multi-Head acelerada en Rust (Soporta GQA + Block-Quant + Modo F32).
    """
    def __init__(self, reader=None, prefix=None, centroids=None, w_q=None, w_k=None, w_v=None, stride=None, n_heads_q=None, n_heads_kv=None, mode='genomic'):
        self.mode = mode
        if reader is not None:
            tensor_q = next(t for t in reader.tensors if t.name == prefix + "attn_q.weight")
            tensor_k = next(t for t in reader.tensors if t.name == prefix + "attn_k.weight")
            
            head_dim = 64
            self.n_heads_q = tensor_q.shape[1] // head_dim
            self.n_heads_kv = tensor_k.shape[1] // head_dim
            
            # Cargar pesos F32 para modo f32
            self.w_q_f32 = dequantize_q8_0(tensor_q.data, tensor_q.shape[1], tensor_q.shape[0])
            self.w_k_f32 = dequantize_q8_0(tensor_k.data, tensor_k.shape[1], tensor_k.shape[0])
            tensor_v = next(t for t in reader.tensors if t.name == prefix + "attn_v.weight")
            self.w_v_f32 = dequantize_q8_0(tensor_v.data, tensor_v.shape[1], tensor_v.shape[0])
            self.head_dim = head_dim

            if mode == 'genomic':
                def get_dna_and_centroids(w_f32, in_f):
                    packed_rows, layer_centroids = [], []
                    for row in w_f32:
                        std = np.std(row); mean = np.mean(row)
                        thresholds = [mean - 0.9816 * std, mean, mean + 0.9816 * std]
                        centroids_row = [mean - 1.510 * std, mean - 0.4528 * std, mean + 0.4528 * std, mean + 1.510 * std]
                        layer_centroids.extend(centroids_row)
                        packed_rows.append(dna_semantic_compression.quantize_embedding(row.tolist(), thresholds))
                    return b"".join(packed_rows), layer_centroids, in_f // 4

                w_q_dna, c_q, stride = get_dna_and_centroids(self.w_q_f32, tensor_q.shape[0])
                w_k_dna, c_k, _ = get_dna_and_centroids(self.w_k_f32, tensor_k.shape[0])
                w_v_dna, c_v, _ = get_dna_and_centroids(self.w_v_f32, tensor_v.shape[0])
                all_centroids = c_q + c_k + c_v
                self.kernel = dna_semantic_compression.GenomicAttention(w_q_dna, w_k_dna, w_v_dna, all_centroids, stride, self.n_heads_q, self.n_heads_kv)
        else:
            # Cargar desde datos persistidos
            self.kernel = dna_semantic_compression.GenomicAttention(w_q, w_k, w_v, centroids.tolist(), stride, n_heads_q, n_heads_kv)

    def forward(self, x, pos):
        if self.mode == 'f32':
            # MHA F32 Simplificado (Teacher proxy)
            q = np.dot(self.w_q_f32, x)
            return q.tolist() # Para el proxy, q es suficiente para activar FFN
            
        return self.kernel.forward(x, pos)

class GenomicTransformerBlock:
    def __init__(self, block_idx, reader=None, input_dir=None, mode='genomic'):
        self.block_idx = block_idx
        self.layers = {}
        self.mode = mode
        
        if reader is not None:
            prefix = f"blk.{block_idx}."
            if mode == 'genomic': print(f"\n🧬 [Bloque {block_idx}] Genomizando con Block-Quant...")
            self.attn = GenomicAttentionLayer(reader, prefix, mode=mode)
            for name in ["ffn_up", "ffn_down"]:
                tensor = next(t for t in reader.tensors if t.name == prefix + name + ".weight")
                self.layers[name] = GenomicLLMLayer(tensor.name, tensor, mode=mode)
        else:
            # ... (Carga desde disco omitida para brevedad, permanece igual)
            # Cargar desde disco
            block_dir = os.path.join(input_dir, f"block_{block_idx}")
            
            # 1. Cargar Atención
            w_q = open(os.path.join(block_dir, "attn_w_q.bin"), "rb").read()
            w_k = open(os.path.join(block_dir, "attn_w_k.bin"), "rb").read()
            w_v = open(os.path.join(block_dir, "attn_w_v.bin"), "rb").read()
            c_attn = np.load(os.path.join(block_dir, "attn_centroids.npy"))
            
            # Inferir dimensiones (provisional para el prototipo)
            # En Qwen2-0.5B: n_q=14, n_kv=2, head_dim=64
            # Stride = 896 / 4 = 224
            self.attn = GenomicAttentionLayer(
                centroids=c_attn, w_q=w_q, w_k=w_k, w_v=w_v, 
                stride=224, n_heads_q=14, n_heads_kv=2
            )
            
            # 2. Cargar FFN
            for name in ["ffn_up", "ffn_down"]:
                db = open(os.path.join(block_dir, f"{name}_db.bin"), "rb").read()
                c = np.load(os.path.join(block_dir, f"{name}_centroids.npy"))
                self.layers[name] = GenomicLLMLayer(name, database=db, centroids=c)

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
    def __init__(self, model_path_or_dir, num_blocks=1, load_genomic=False, mode='genomic'):
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
        self.mode = mode
        
        if load_genomic:
            self.load_genomic_model(model_path_or_dir)
        else:
            self.reader = gguf.GGUFReader(model_path_or_dir)
            print(f"\n[*] Inicializando Modelo (Modo: {mode})...")
            
            embd_tensor = next(t for t in self.reader.tensors if t.name == "token_embd.weight")
            self.embedding_matrix = dequantize_q8_0(embd_tensor.data, embd_tensor.shape[1], embd_tensor.shape[0])
            
            print("[*] Cargando RMSNorm final...")
            self.output_norm_weight = next(t for t in self.reader.tensors if t.name == "output_norm.weight").data.astype(np.float32)
            
            self.blocks = [GenomicTransformerBlock(i, reader=self.reader, mode=mode) for i in range(num_blocks)]

    def save_genomic_model(self, output_dir):
        """
        Guarda el modelo genomizado completo en un directorio.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        print(f"[*] Exportando modelo genómico a: {output_dir}...")
        
        np.save(os.path.join(output_dir, "embedding_matrix.npy"), self.embedding_matrix)
        np.save(os.path.join(output_dir, "output_norm.npy"), self.output_norm_weight)
        
        for i, block in enumerate(self.blocks):
            block_dir = os.path.join(output_dir, f"block_{i}")
            if not os.path.exists(block_dir): os.makedirs(block_dir)
            
            # Guardar Atención
            with open(os.path.join(block_dir, "attn_w_q.bin"), "wb") as f: f.write(block.attn.kernel.w_q)
            with open(os.path.join(block_dir, "attn_w_k.bin"), "wb") as f: f.write(block.attn.kernel.w_k)
            with open(os.path.join(block_dir, "attn_w_v.bin"), "wb") as f: f.write(block.attn.kernel.w_v)
            np.save(os.path.join(block_dir, "attn_centroids.npy"), np.array(block.attn.kernel.centroids))
            
            # Guardar FFN
            for name, layer in block.layers.items():
                with open(os.path.join(block_dir, f"{name}_db.bin"), "wb") as f: f.write(layer.engine.database)
                np.save(os.path.join(block_dir, f"{name}_centroids.npy"), np.array(layer.engine.centroids))

    def load_genomic_model(self, input_dir):
        print(f"[*] Cargando modelo genómico desde: {input_dir}...")
        self.embedding_matrix = np.load(os.path.join(input_dir, "embedding_matrix.npy"))
        self.output_norm_weight = np.load(os.path.join(input_dir, "output_norm.npy"))
        
        block_dirs = sorted([d for d in os.listdir(input_dir) if d.startswith("block_")], key=lambda x: int(x.split("_")[1]))
        self.blocks = []
        for b_dir in block_dirs:
            idx = int(b_dir.split("_")[1])
            self.blocks.append(GenomicTransformerBlock(idx, input_dir=input_dir))

    def rms_norm(self, x, weight, eps=1e-6):
        x = np.array(x)
        rms = np.sqrt(np.mean(x**2) + eps)
        return (x / rms) * weight

    def generate(self, prompt, max_new_tokens=20, temperature=0.8, top_p=0.9, repetition_penalty=1.1):
        input_ids = self.tokenizer.encode(prompt)
        print(f"\n🚀 Generando (Temp={temperature}, Top-P={top_p}): '{prompt}'")
        
        generated_ids = []
        for i in range(max_new_tokens):
            last_id = input_ids[-1]
            x = self.embedding_matrix[last_id].tolist()
            
            for j, block in enumerate(self.blocks):
                x = block.forward(x, len(input_ids) - 1)
            
            # Final Normalization
            x = self.rms_norm(x, self.output_norm_weight)
            
            # Output Head + Sampling
            logits = np.dot(self.embedding_matrix, x).tolist()
            logits = dna_semantic_compression.apply_repetition_penalty(
                logits, input_ids, repetition_penalty
            )
            
            next_id = dna_semantic_compression.sample_top_p(logits, temperature, top_p)
            
            if next_id == self.tokenizer.eos_token_id: break
            
            generated_ids.append(next_id)
            input_ids.append(next_id)
            
            word = self.tokenizer.decode([next_id])
            print(word, end="", flush=True)
            
        return self.tokenizer.decode(generated_ids)

if __name__ == "__main__":
    model_path = "/data/data/com.termux/files/home/models/qwen2-0_5b-q8_0.gguf"
    # Cargamos 12 bloques para balancear velocidad y calidad en este test
    model = GenomicLLM(model_path, num_blocks=12)
    model.generate("El protocolo GAJE es", max_new_tokens=50, temperature=0.7)
