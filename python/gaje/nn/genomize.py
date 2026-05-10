import gguf
import numpy as np
from gaje.core import _impl as dna_semantic_compression
import os
from transformers import AutoTokenizer

def dequantize_q8_0(tensor, n_head=None, head_dim=None, is_q_or_k=False):
    in_features, out_features = tensor.shape
    data = tensor.data.tobytes()
    flat_weights = dna_semantic_compression.dequantize_q8_0_native(data, out_features, in_features)
    w = np.array(flat_weights, dtype=np.float32).reshape(out_features, in_features)
    
    # Des-permutación para Llama 3 / Qwen2 (GGUF)
    # Llama.cpp permuta Q y K para que el RoPE split sea contiguo en memoria.
    if is_q_or_k and n_head is not None and head_dim is not None:
        w_new = np.zeros_like(w)
        for h in range(n_head):
            for i in range(head_dim // 2):
                w_new[h * head_dim + i] = w[h * head_dim + 2 * i]
                w_new[h * head_dim + head_dim // 2 + i] = w[h * head_dim + 2 * i + 1]
        return w_new
    return w

class GenomicLLM:
    def __init__(self, model_path):
        print("DEBUG: Using GenomicLLM from genomize.py (RoPE Split + De-permutation)")
        print(f"🧬 Sincronizando Organismo Completo: {os.path.basename(model_path)}")
        self.reader = gguf.GGUFReader(model_path)
        
        # Detect architecture
        if "general.architecture" in self.reader.fields:
            part = self.reader.fields["general.architecture"].parts[-1]
            if isinstance(part[0], (bytes, bytearray)):
                arch = part[0].decode("utf-8")
            else:
                arch = bytes(part).decode("utf-8")
        else:
            arch = "llama"
        print(f"[*] Arquitectura detectada: {arch}")

        self.n_embd = int(self.reader.fields[f"{arch}.embedding_length"].parts[-1][0])
        self.n_head = int(self.reader.fields[f"{arch}.attention.head_count"].parts[-1][0])
        self.n_head_kv = int(self.reader.fields[f"{arch}.attention.head_count_kv"].parts[-1][0])
        self.head_dim = self.n_embd // self.n_head
        self.n_blocks = int(self.reader.fields[f"{arch}.block_count"].parts[-1][0])
        self.rope_base = float(self.reader.fields[f"{arch}.rope.freq_base"].parts[-1][0])
        self.eps = float(self.reader.fields[f"{arch}.attention.layer_norm_rms_epsilon"].parts[-1][0])
        
        self.token_embd = dequantize_q8_0(next(t for t in self.reader.tensors if t.name == "token_embd.weight"))
        self.output_norm = next(t for t in self.reader.tensors if t.name == "output_norm.weight").data.astype(np.float32)
        self.output_weight = self.token_embd 

        self.blocks = []
        for i in range(self.n_blocks):
            self.blocks.append(TransformerBlock(self.reader, i, self.n_head, self.n_head_kv, self.head_dim, self.rope_base, self.eps))
            if (i+1) % 10 == 0: print(f"    [~] Bloque {i+1}/{self.n_blocks} sincronizado...")
        
        # Select correct tokenizer
        tokenizer_name = "Qwen/Qwen2-0.5B" if arch == "qwen2" else "HuggingFaceTB/SmolLM2-135M-Instruct"
        print(f"[*] Cargando tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def rms_norm(self, x, weight):
        rms = np.sqrt(np.mean(x**2) + self.eps)
        return (x / rms) * weight

    def clear_cache(self):
        for b in self.blocks: b.attn.clear_cache()

    def forward(self, tokens):
        self.clear_cache()
        h = None
        for i, tid in enumerate(tokens):
            h = self.token_embd[tid].copy()
            for block in self.blocks:
                h = block.forward(h, i)
        h = self.rms_norm(h, self.output_norm)
        logits = np.dot(self.output_weight, h)
        return logits

class TransformerBlock:
    def __init__(self, reader, idx, n_head, n_head_kv, head_dim, rope_base, eps):
        self.idx = idx
        self.eps = eps
        p = f"blk.{idx}."
        self.attn_norm = next(t for t in reader.tensors if t.name == p + "attn_norm.weight").data.astype(np.float32)
        self.ffn_norm = next(t for t in reader.tensors if t.name == p + "ffn_norm.weight").data.astype(np.float32)
        self.attn = AttentionLayer(reader, p, n_head, n_head_kv, head_dim, rope_base)
        self.w_gate = dequantize_q8_0(next(t for t in reader.tensors if t.name == p + "ffn_gate.weight"))
        self.w_up = dequantize_q8_0(next(t for t in reader.tensors if t.name == p + "ffn_up.weight"))
        self.w_down = dequantize_q8_0(next(t for t in reader.tensors if t.name == p + "ffn_down.weight"))

    def rms_norm(self, x, weight):
        rms = np.sqrt(np.mean(x**2) + self.eps)
        return (x / rms) * weight

    def silu(self, x):
        return x * (1.0 / (1.0 + np.exp(-np.clip(x, -20, 20))))

    def forward(self, x, pos):
        h = self.rms_norm(x, self.attn_norm)
        h = self.attn.forward(h, pos)
        x = x + h
        h = self.rms_norm(x, self.ffn_norm)
        gate = np.dot(self.w_gate, h)
        up = np.dot(self.w_up, h)
        h = self.silu(gate) * up
        h = np.dot(self.w_down, h)
        x = x + h
        return x

class AttentionLayer:
    def __init__(self, reader, p, n_head, n_head_kv, head_dim, rope_base):
        self.n_head = n_head
        self.n_head_kv = n_head_kv
        self.head_dim = head_dim
        self.rope_base = rope_base
        self.k_cache = []
        self.v_cache = []
        self.w_q = dequantize_q8_0(next(t for t in reader.tensors if t.name == p + "attn_q.weight"), n_head, head_dim, True)
        self.w_k = dequantize_q8_0(next(t for t in reader.tensors if t.name == p + "attn_k.weight"), n_head_kv, head_dim, True)
        self.w_v = dequantize_q8_0(next(t for t in reader.tensors if t.name == p + "attn_v.weight"))
        self.w_o = dequantize_q8_0(next(t for t in reader.tensors if t.name == p + "attn_output.weight"))

    def clear_cache(self):
        self.k_cache = []
        self.v_cache = []

    def apply_rope(self, x, pos):
        n_heads, dim = x.shape
        res = np.zeros_like(x)
        inv_freq = 1.0 / (self.rope_base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        theta = pos * inv_freq
        cos = np.cos(theta); sin = np.sin(theta)
        for h in range(n_heads):
            v0 = x[h, :dim//2]; v1 = x[h, dim//2:]
            res[h, :dim//2] = v0 * cos - v1 * sin
            res[h, dim//2:] = v0 * sin + v1 * cos
        return res

    def forward(self, x, pos):
        q = np.dot(self.w_q, x).reshape(self.n_head, self.head_dim)
        k = np.dot(self.w_k, x).reshape(self.n_head_kv, self.head_dim)
        v = np.dot(self.w_v, x).reshape(self.n_head_kv, self.head_dim)
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
        return np.dot(self.w_o, attn_out)
