import os
import numpy as np
import dna_semantic_compression
import gguf
import time
from transformers import AutoTokenizer

def dequantize_q8_0(tensor, n_head=None, head_dim=None, is_q_or_k=False):
    in_features, out_features = tensor.shape
    data = tensor.data.tobytes()
    flat_weights = dna_semantic_compression.dequantize_q8_0_native(data, out_features, in_features)
    w = np.array(flat_weights, dtype=np.float32).reshape(out_features, in_features)
    
    if is_q_or_k and n_head is not None and head_dim is not None:
        # Reshape to (n_head, head_dim, in_features)
        w = w.reshape(n_head, head_dim, in_features)
        # Qwen2/Llama style RoPE de-permutation:
        # Standard GGML/GGUF for Qwen2 often keeps [0, 1, 2, ..., d/2-1, d/2, ..., d-1]
        # and RoPE is applied as (x[i], x[i+d/2])
        w_new = np.zeros_like(w)
        for h in range(n_head):
            for i in range(head_dim // 2):
                # We want our interleaved RoPE [v0, v1] to pick [w[i], w[i+d/2]]
                w_new[h, 2 * i] = w[h, i]
                w_new[h, 2 * i + 1] = w[h, i + head_dim // 2]
        return w_new.reshape(out_features, in_features)
    return w

class GenomicLayer:
    def __init__(self, name, weights_f32, block_size=32):
        self.name = name
        self.out_features, self.in_features = weights_f32.shape
        self.block_size = block_size
        
        # Ensure in_features is multiple of block_size
        reshaped = weights_f32.reshape(-1, block_size)
        stds = np.std(reshaped, axis=1, keepdims=True)
        means = np.mean(reshaped, axis=1, keepdims=True)
        
        base_centroids = np.array([-1.510, -0.4528, 0.4528, 1.510], dtype=np.float32)
        block_centroids = (means + base_centroids * stds).astype(np.float32)
        
        base_thresholds = np.array([-1.0, 0.0, 1.0])
        
        dna_batch = []
        for i in range(len(reshaped)):
            # Per-block thresholding: center at mean, spread by std
            block_mean = means[i][0]
            block_std = stds[i][0]
            t = (block_mean + base_thresholds * block_std).tolist()
            
            dna = dna_semantic_compression.quantize_embedding(reshaped[i].tolist(), t)
            dna_batch.append(dna)
            
        self.linear = dna_semantic_compression.GenomicLinear(
            b"".join(dna_batch),
            block_centroids.flatten().tolist(),
            self.out_features,
            self.in_features,
            self.block_size
        )

    def forward(self, x):
        return np.array(self.linear.forward(x.tolist()), dtype=np.float32)
        
    def get_row(self, idx):
        n_blocks = self.in_features // self.block_size
        stride = self.block_size // 4
        row_offset = idx * n_blocks * stride
        dna_row = self.linear.database[row_offset : row_offset + n_blocks * stride]
        
        # Dequantize block-by-block
        res = np.zeros(self.in_features, dtype=np.float32)
        for b in range(n_blocks):
            dna_block = dna_row[b*stride : (b+1)*stride]
            c_offset = (idx * n_blocks + b) * 4
            c = self.linear.centroids[c_offset : c_offset + 4]
            res[b*self.block_size : (b+1)*self.block_size] = dna_semantic_compression.dequantize_embedding(dna_block, self.block_size, c)
        return res

class GenomicAttentionLayer:
    def __init__(self, reader, p, n_head, n_head_kv, head_dim, rope_base):
        self.n_head = n_head
        self.n_head_kv = n_head_kv
        self.head_dim = head_dim
        self.rope_base = rope_base
        self.k_cache = []
        self.v_cache = []
        
        w_q_f32 = dequantize_q8_0(next(t for t in reader.tensors if t.name == p + "attn_q.weight"), n_head, head_dim, True)
        w_k_f32 = dequantize_q8_0(next(t for t in reader.tensors if t.name == p + "attn_k.weight"), n_head_kv, head_dim, True)
        w_v_f32 = dequantize_q8_0(next(t for t in reader.tensors if t.name == p + "attn_v.weight"))
        w_o_f32 = dequantize_q8_0(next(t for t in reader.tensors if t.name == p + "attn_output.weight"))
        
        self.w_q = GenomicLayer(p + "attn_q", w_q_f32)
        self.w_k = GenomicLayer(p + "attn_k", w_k_f32)
        self.w_v = GenomicLayer(p + "attn_v", w_v_f32)
        self.w_o = GenomicLayer(p + "attn_output", w_o_f32)

    def clear_cache(self):
        self.k_cache = []
        self.v_cache = []

    def apply_rope(self, x, pos):
        # x is (n_heads, head_dim)
        n_heads, dim = x.shape
        # Create frequencies for RoPE
        inv_freq = 1.0 / (self.rope_base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        t = np.array([pos], dtype=np.float32)
        freqs = np.outer(t, inv_freq).flatten() # (dim/2,)
        
        # Interleaved RoPE: [x0, x1, x2, x3] -> [x0*cos-x1*sin, x0*sin+x1*cos, x2*cos-x3*sin, x2*sin+x3*cos]
        cos = np.cos(freqs)
        sin = np.sin(freqs)
        
        res = np.zeros_like(x)
        for h in range(n_heads):
            x_h = x[h]
            # Even indices: 0, 2, 4...
            x_even = x_h[0::2]
            # Odd indices: 1, 3, 5...
            x_odd = x_h[1::2]
            
            res[h, 0::2] = x_even * cos - x_odd * sin
            res[h, 1::2] = x_even * sin + x_odd * cos
        return res

    def forward(self, x, pos):
        # x is (576,)
        q_raw = self.w_q.forward(x)
        k_raw = self.w_k.forward(x)
        v_raw = self.w_v.forward(x)
        
        if q_raw.shape[0] != self.n_head * self.head_dim:
             print(f"DEBUG Q SHAPE: {q_raw.shape}, expected {self.n_head * self.head_dim}")
             
        q = q_raw.reshape(self.n_head, self.head_dim)
        k = k_raw.reshape(self.n_head_kv, self.head_dim)
        v = v_raw.reshape(self.n_head_kv, self.head_dim)
        q = self.apply_rope(q, pos); k = self.apply_rope(k, pos)
        self.k_cache.append(k); self.v_cache.append(v)
        k_full = np.stack(self.k_cache); v_full = np.stack(self.v_cache)
        if self.n_head != self.n_head_kv:
            reps = self.n_head // self.n_head_kv
            k_full = np.repeat(k_full, reps, axis=1)
            v_full = np.repeat(v_full, reps, axis=1)
        scale = 1.0 / np.sqrt(self.head_dim)
        scores = np.einsum('hd,shd->hs', q, k_full) * scale
        probs = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        probs /= (np.sum(probs, axis=-1, keepdims=True) + 1e-9)
        attn_out = np.einsum('hs,shd->hd', probs, v_full).flatten()
        return self.w_o.forward(attn_out)

class GenomicTransformerBlock:
    def __init__(self, reader, idx, n_head, n_head_kv, head_dim, rope_base, eps):
        self.idx = idx
        self.eps = eps
        p = f"blk.{idx}."
        self.attn_norm = next(t for t in reader.tensors if t.name == p + "attn_norm.weight").data.astype(np.float32)
        self.ffn_norm = next(t for t in reader.tensors if t.name == p + "ffn_norm.weight").data.astype(np.float32)
        self.attn = GenomicAttentionLayer(reader, p, n_head, n_head_kv, head_dim, rope_base)
        
        w_gate_f32 = dequantize_q8_0(next(t for t in reader.tensors if t.name == p + "ffn_gate.weight"))
        w_up_f32 = dequantize_q8_0(next(t for t in reader.tensors if t.name == p + "ffn_up.weight"))
        w_down_f32 = dequantize_q8_0(next(t for t in reader.tensors if t.name == p + "ffn_down.weight"))
        
        self.w_gate = GenomicLayer(p + "ffn_gate", w_gate_f32)
        self.w_up = GenomicLayer(p + "ffn_up", w_up_f32)
        self.w_down = GenomicLayer(p + "ffn_down", w_down_f32)

    def rms_norm(self, x, weight):
        rms = np.sqrt(np.mean(x**2) + self.eps)
        return (x / rms) * weight

    def silu(self, x):
        return x * (1.0 / (1.0 + np.exp(-np.clip(x, -20, 20))))

    def forward(self, x, pos):
        # x is (576,)
        h = self.rms_norm(x, self.attn_norm)
        h = self.attn.forward(h, pos) # h is (576,)
        x = x + h
        h = self.rms_norm(x, self.ffn_norm)
        gate = self.w_gate.forward(h)
        up = self.w_up.forward(h)
        h = self.silu(gate) * up
        h = self.w_down.forward(h)
        x = x + h
        return x

class GenomicLLM:
    def __init__(self, model_path, num_blocks=None):
        print(f"🧬 Sincronizando Organismo Genómico (2-bit): {os.path.basename(model_path)}")
        self.reader = gguf.GGUFReader(model_path)
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
        
        w_embd_f32 = dequantize_q8_0(next(t for t in self.reader.tensors if t.name == "token_embd.weight"))
        self.embeddings = GenomicLayer("token_embd", w_embd_f32)
        
        self.output_norm = next(t for t in self.reader.tensors if t.name == "output_norm.weight").data.astype(np.float32)
        
        try:
            w_head_f32 = dequantize_q8_0(next(t for t in self.reader.tensors if t.name == "output.weight"))
        except StopIteration:
            w_head_f32 = w_embd_f32
            
        self.lm_head = GenomicLayer("lm_head", w_head_f32)

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
