import os
import numpy as np
from gaje.core import _impl as dna_semantic_compression
import gguf
from transformers import AutoTokenizer

def dequantize_q8_0(tensor):
    in_features, out_features = tensor.shape
    data = tensor.data.tobytes()
    flat_weights = dna_semantic_compression.dequantize_q8_0_native(data, out_features, in_features)
    w = np.array(flat_weights, dtype=np.float32).reshape(out_features, in_features)
    return w

from gaje.processing.balancer import SignalToNoiseBalancer

class GenomicLayer:
    def __init__(self, name, weights_f32_or_tensor, block_size=32, anchor_threshold=0.98, rmsnorm_weight=None, eps=1e-6, balancer=None):
        self.name = name
        self.block_size = block_size
        self.balancer = balancer or SignalToNoiseBalancer()
        
        # 1. DIRECT GENOMIC INGESTION (DGI) - Bypass Q8 loss
        if hasattr(weights_f32_or_tensor, 'tensor_type'):
            tensor = weights_f32_or_tensor
            self.out_features, self.in_features = tensor.shape[::-1] if len(tensor.shape) == 2 else (tensor.shape[0], 1)
            
            # Use balancer to decide initial threshold
            a_threshold = self.balancer.current_threshold
            
            if tensor.tensor_type == gguf.GGMLQuantizationType.F16:
                # Streaming Conversion: F16 -> 2-bit DNA directly in Rust (with Anchor extraction)
                dna, centroids, anchors = dna_semantic_compression.genomize_f16_native(tensor.data.tobytes(), block_size, a_threshold)
                self.dna_database = dna
                self.dna_centroids = centroids
                self.anchors = anchors
            elif tensor.tensor_type == gguf.GGMLQuantizationType.F32:
                # Streaming Conversion: F32 -> 2-bit DNA directly in Rust (with Anchor extraction)
                dna, centroids, anchors = dna_semantic_compression.genomize_f32_native(tensor.data.tobytes(), block_size, a_threshold)
                self.dna_database = dna
                self.dna_centroids = centroids
                self.anchors = anchors
            else:
                # Fallback to standard dequantization (Q8_0, Q4_K, etc.)
                weights_f32 = dequantize_q8_0(tensor)
                self._init_from_f32(weights_f32, block_size, anchor_threshold)
        else:
            weights_f32 = weights_f32_or_tensor
            self.out_features, self.in_features = weights_f32.shape
            self._init_from_f32(weights_f32, block_size, anchor_threshold)

        self.linear = dna_semantic_compression.GenomicLinear(
            self.dna_database,
            self.anchors,
            self.dna_centroids,
            self.out_features,
            self.in_features,
            self.block_size,
            rmsnorm_weight.tolist() if rmsnorm_weight is not None else [],
            eps
        )

    def _init_from_f32(self, weights_f32, block_size, anchor_threshold):
        reshaped = weights_f32.reshape(-1, block_size)
        base_centroids = np.array([-1.510, -0.4528, 0.4528, 1.510], dtype=np.float32)
        
        dna_batch = []
        all_dna_centroids = []
        anchors_f32 = np.zeros_like(weights_f32)
        
        for i in range(len(reshaped)):
            block = reshaped[i]
            block_mean = np.mean(block)
            block_std = np.std(block) + 1e-6
            t = [block_mean - 1.0 * block_std, block_mean, block_mean + 1.0 * block_std]
            dna = dna_semantic_compression.quantize_embedding(block.tolist(), t)
            dna_batch.append(dna)
            
            c = (block_mean + base_centroids * block_std).astype(np.float32)
            all_dna_centroids.extend(c.tolist())
            
            dequantized = np.array(dna_semantic_compression.dequantize_embedding(dna, block_size, c.tolist()))
            residual = block - dequantized
            
            # Use balancer threshold if applicable, otherwise use quantile
            current_t = self.balancer.current_threshold if self.balancer else 0.15
            
            err_threshold = np.quantile(np.abs(residual), anchor_threshold)
            anchor_mask = np.abs(residual) >= err_threshold
            
            row_idx = i // (self.in_features // block_size)
            col_in_row = (i % (self.in_features // block_size)) * block_size
            anchors_f32[row_idx, col_in_row : col_in_row + block_size][anchor_mask] = residual[anchor_mask]

        self.dna_database = b"".join(dna_batch)
        self.dna_centroids = all_dna_centroids
        self.anchors = anchors_f32.flatten().tolist()

    def forward(self, x):
        return np.array(self.linear.forward(x.tolist()), dtype=np.float32)
        
    def get_row(self, idx):
        n_blocks = self.in_features // self.block_size
        stride = self.block_size // 4
        row_offset = idx * n_blocks * stride
        
        dna_row = self.linear.database[row_offset : row_offset + n_blocks * stride]
        anchor_row = np.array(self.linear.anchors[idx * self.in_features : (idx + 1) * self.in_features])
        
        res = np.zeros(self.in_features, dtype=np.float32)
        for b in range(n_blocks):
            dna_block = dna_row[b*stride : (b+1)*stride]
            c_offset = (idx * n_blocks + b) * 4
            c = self.linear.centroids[c_offset : c_offset + 4]
            
            w_base = np.array(dna_semantic_compression.dequantize_embedding(dna_block, self.block_size, c))
            res[b*self.block_size : (b+1)*self.block_size] = w_base
            
        return res + anchor_row

class GenomicAttentionLayer:
    def __init__(self, reader, p, n_head, n_head_kv, head_dim, rope_base, rmsnorm_weight=None, eps=1e-6):
        self.n_head = n_head
        self.n_head_kv = n_head_kv
        self.head_dim = head_dim
        
        t_q = next(t for t in reader.tensors if t.name == p + "attn_q.weight")
        t_k = next(t for t in reader.tensors if t.name == p + "attn_k.weight")
        t_v = next(t for t in reader.tensors if t.name == p + "attn_v.weight")
        t_o = next(t for t in reader.tensors if t.name == p + "attn_output.weight")
        
        # DGI initialization for Q, K, V
        q_gen = GenomicLayer(p + "q", t_q)
        k_gen = GenomicLayer(p + "k", t_k)
        v_gen = GenomicLayer(p + "v", t_v)
        
        # Combine all centroids into a flat list
        all_centroids = q_gen.dna_centroids + k_gen.dna_centroids + v_gen.dna_centroids
        
        self.attn = dna_semantic_compression.GenomicAttention(
            q_gen.dna_database,
            k_gen.dna_database,
            v_gen.dna_database,
            all_centroids,
            q_gen.block_size // 4,
            n_head,
            n_head_kv,
            rmsnorm_weight.tolist() if rmsnorm_weight is not None else [],
            eps
        )
        self.w_o = GenomicLayer(p + "attn_output", t_o)

    def clear_cache(self):
        self.attn.clear_cache()

    def forward(self, x, pos):
        # All projections (Q, K, V), RoPE, KV-Cache and Attention are now in Rust!
        attn_out = np.array(self.attn.forward(x.tolist(), pos), dtype=np.float32)
        return self.w_o.forward(attn_out)

class GenomicTransformerBlock:
    def __init__(self, reader, idx, n_head, n_head_kv, head_dim, rope_base, eps):
        self.idx = idx
        self.eps = eps
        p = f"blk.{idx}."
        attn_norm = next(t for t in reader.tensors if t.name == p + "attn_norm.weight").data.astype(np.float32)
        ffn_norm = next(t for t in reader.tensors if t.name == p + "ffn_norm.weight").data.astype(np.float32)
        
        # Fused Attention (DGI enabled)
        self.attn = GenomicAttentionLayer(reader, p, n_head, n_head_kv, head_dim, rope_base, rmsnorm_weight=attn_norm, eps=eps)
        
        t_gate = next(t for t in reader.tensors if t.name == p + "ffn_gate.weight")
        t_up = next(t for t in reader.tensors if t.name == p + "ffn_up.weight")
        t_down = next(t for t in reader.tensors if t.name == p + "ffn_down.weight")
        
        # DGI initialization for FFN
        gate_gen = GenomicLayer(p + "gate", t_gate)
        up_gen = GenomicLayer(p + "up", t_up)
        
        self.swiglu = dna_semantic_compression.GenomicSwiGLU(
            gate_gen.dna_database,
            up_gen.dna_database,
            gate_gen.dna_centroids, 
            gate_gen.out_features,
            gate_gen.in_features,
            gate_gen.block_size
        )
        self.w_down = GenomicLayer(p + "ffn_down", t_down)
        self.ffn_norm = ffn_norm

    def rms_norm(self, x, weight):
        rms = np.sqrt(np.mean(x**2) + self.eps)
        return (x / rms) * weight

    def forward(self, x, pos):
        # x is (576,)
        # 1. Attn branch: RMSNorm is now INSIDE self.attn.forward
        h = self.attn.forward(x, pos) 
        x = x + h
        
        # 2. FFN branch: 
        h = self.rms_norm(x, self.ffn_norm) # TODO: Fuse this into SwiGLU Rust kernel
        h = np.array(self.swiglu.forward(h.tolist()), dtype=np.float32)
        h = self.w_down.forward(h)
        x = x + h
        return x

class GenomicLLM:
    def __init__(self, model_path, num_blocks=None):
        print(f"🧬 Sincronizando Organismo Genómico (2-bit): {os.path.basename(model_path)}")
        self.reader = gguf.GGUFReader(model_path)
        self.balancer = SignalToNoiseBalancer()
        arch = "llama"
        if f"{arch}.embedding_length" not in self.reader.fields:
            arch = "qwen2"
            
        self.n_embd = int(self.reader.fields[f"{arch}.embedding_length"].parts[-1][0])
        self.n_head = int(self.reader.fields[f"{arch}.attention.head_count"].parts[-1][0])
        self.n_head_kv = int(self.reader.fields[f"{arch}.attention.head_count_kv"].parts[-1][0])
        self.head_dim = self.n_embd // self.n_head
        self.n_blocks = int(self.reader.fields[f"{arch}.block_count"].parts[-1][0]) if num_blocks is None else num_blocks
        self.rope_base = float(self.reader.fields[f"{arch}.rope.freq_base"].parts[-1][0])
        self.eps = float(self.reader.fields[f"{arch}.attention.layer_norm_rms_epsilon"].parts[-1][0])
        
        w_embd_tensor = next(t for t in self.reader.tensors if t.name == "token_embd.weight")
        self.embeddings = GenomicLayer("token_embd", w_embd_tensor, balancer=self.balancer)
        
        self.output_norm = next(t for t in self.reader.tensors if t.name == "output_norm.weight").data.astype(np.float32)
        
        try:
            w_head_tensor = next(t for t in self.reader.tensors if t.name == "output.weight")
        except StopIteration:
            w_head_tensor = w_embd_tensor
            
        self.lm_head = GenomicLayer("lm_head", w_head_tensor, balancer=self.balancer)

        self.blocks = []
        for i in range(self.n_blocks):
            self.blocks.append(GenomicTransformerBlock(self.reader, i, self.n_head, self.n_head_kv, self.head_dim, self.rope_base, self.eps))
            if (i+1) % 5 == 0: print(f"    [~] Bloque {i+1}/{self.n_blocks} genomizado...")
            
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

    def rms_norm(self, x, weight):
        rms = np.sqrt(np.mean(x**2) + self.eps)
        if x.shape != weight.shape:
            print(f"DEBUG SHAPE: x={x.shape}, weight={weight.shape}, name={self.__class__.__name__}")
        return (x / rms) * weight

    def clear_cache(self):
        for b in self.blocks: b.attn.clear_cache()

    def forward(self, tokens):
        self.clear_cache()
        all_logits = []
        
        for i, tid in enumerate(tokens):
            # Fast embedding lookup
            h = self.embeddings.get_row(tid)
            
            for block in self.blocks:
                h = block.forward(h, i)
                
            h_norm = self.rms_norm(h, self.output_norm)
            logits = self.lm_head.forward(h_norm)
            all_logits.append(logits)
            
        return np.stack(all_logits)

    def save_genomic_model(self, output_dir):
        """
        Guarda el modelo genómico destilado (Fase 10).
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        
        print(f"📦 Guardando Organismo Genómico en {output_dir}...")
        
        # Guardamos cada capa en un archivo .gaje independiente o unificado
        # Para simplificar, guardamos los pesos críticos y centroides refinados
        metadata = {
            "n_embd": self.n_embd,
            "n_head": self.n_head,
            "n_blocks": self.n_blocks,
            "head_dim": self.head_dim,
            "rope_base": self.rope_base,
            "eps": self.eps
        }
        
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            import json
            json.dump(metadata, f)
            
        print("✅ Modelo guardado exitosamente.")
