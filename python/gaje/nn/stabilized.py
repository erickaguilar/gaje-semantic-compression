import os
import numpy as np
from gaje.core import _impl as dna_semantic_compression
import gguf
import time
from transformers import AutoTokenizer

from gaje.utils.quantization import dequantize_q8_0
from gaje.processing.balancer import SignalToNoiseBalancer

class TensorLoader:
    """Utility class to index and retrieve tensors from a GGUF reader efficiently."""
    def __init__(self, reader):
        self.tensors = {t.name: t for t in reader.tensors}
        
    def get(self, name, required=True):
        if name in self.tensors:
            return self.tensors[name]
        if required:
            raise KeyError(f"Required tensor '{name}' not found in GGUF file.")
        return None

class GenomicLayer:
    def __init__(self, name, weights_f32_or_tensor, block_size=32, anchor_threshold=0.98, rmsnorm_weight=None, eps=1e-6, balancer=None, n_head=None, head_dim=None):
        self.name = name
        self.block_size = block_size
        self.balancer = balancer
        is_q_or_k = "attn_q" in name or "attn_k" in name
        
        if hasattr(weights_f32_or_tensor, 'tensor_type'):
            tensor = weights_f32_or_tensor
            self.out_features, self.in_features = tensor.shape[::-1] if len(tensor.shape) == 2 else (tensor.shape[0], 1)
            
            # Determinamos si el tensor necesita des-permutación (Llama/Qwen style)
            # Solo se aplica a las proyecciones Q y K.
            if tensor.tensor_type in [gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32]:
                if tensor.tensor_type == gguf.GGMLQuantizationType.F16:
                    weights_f32 = np.frombuffer(tensor.data, dtype=np.float16).astype(np.float32)
                else:
                    weights_f32 = np.frombuffer(tensor.data, dtype=np.float32)
                
                # Para F16/F32 NO aplicamos des-permutación, ya que suelen venir alineados
                self._init_from_f32(weights_f32, block_size, anchor_threshold)
            else:
                # Solo para Q8_0 aplicamos la lógica de des-permutación necesaria
                weights_f32 = dequantize_q8_0(tensor, n_head, head_dim, is_q_or_k)
                self._init_from_f32(weights_f32, block_size, anchor_threshold)
        else:
            weights_f32 = weights_f32_or_tensor
            self.out_features, self.in_features = weights_f32.shape
            self._init_from_f32(weights_f32, block_size, anchor_threshold)

        self.linear = dna_semantic_compression.GenomicLinear(
            self.dna_database, self.anchors, self.dna_centroids, self.out_features, self.in_features, self.block_size,
            rmsnorm_weight.tolist() if rmsnorm_weight is not None else [], eps, self.precision_mask,
            self.epigenetic_database, self.epigenetic_centroids, self.triplet_database, self.triplet_centroids
        )

    def _init_from_f32(self, weights_f32, block_size, anchor_threshold):
        w_matrix = weights_f32.reshape(self.out_features, self.in_features)
        
        dna_db, dna_centroids, anchors = dna_semantic_compression.genomize_f32_native(
            weights_f32.tobytes(), 
            block_size, 
            anchor_threshold
        )
        
        self.dna_database = dna_db
        self.dna_centroids = dna_centroids
        self.anchors = anchors
        
        self.epigenetic_database = b""
        self.epigenetic_centroids = []
        self.triplet_database = b""
        self.triplet_centroids = []
        
        if self.balancer:
            stride = block_size // 4
            entropies = np.array(dna_semantic_compression.calculate_shannon_entropy(w_matrix.tolist()))
            full_mask = self.balancer.generate_precision_mask(entropies)
            self.precision_mask = [int(np.max(full_mask[b*block_size+k*4 : b*block_size+(k+1)*4])) 
                                   for b in range(self.in_features // block_size) 
                                   for k in range(stride)] * self.out_features
            
            self.epigenetic_database = b"\x00" * (len(dna_db))
            self.epigenetic_centroids = [0.0] * (len(dna_centroids))
            self.triplet_database = b"\x00" * (len(dna_db))
            self.triplet_centroids = [0.0] * (len(dna_centroids))
        else:
            self.precision_mask = []

    def forward(self, x):
        return np.array(self.linear.forward(x.tolist()), dtype=np.float32)
        
    def get_row(self, idx):
        n_blocks, stride = self.in_features // self.block_size, self.block_size // 4
        dna_row = self.dna_database[idx * n_blocks * stride : (idx + 1) * n_blocks * stride]
        res = np.zeros(self.in_features, dtype=np.float32)
        for b in range(n_blocks):
            res[b*self.block_size : (b+1)*self.block_size] = np.array(dna_semantic_compression.dequantize_embedding(dna_row[b*stride : (b+1)*stride], self.block_size, self.dna_centroids[(idx * n_blocks + b) * 4 : (idx * n_blocks + b) * 4 + 4]))
        return res + np.array(self.anchors[idx * self.in_features : (idx + 1) * self.in_features], dtype=np.float32)

class GenomicAttentionLayer:
    def __init__(self, loader, p, n_head, n_head_kv, head_dim, rope_base, rmsnorm_weight=None, eps=1e-6):
        self.q_gen = GenomicLayer(p + "attn_q", loader.get(p + "attn_q.weight"), n_head=n_head, head_dim=head_dim)
        self.k_gen = GenomicLayer(p + "attn_k", loader.get(p + "attn_k.weight"), n_head=n_head_kv, head_dim=head_dim)
        self.v_gen = GenomicLayer(p + "attn_v", loader.get(p + "attn_v.weight"))
        self.w_o = GenomicLayer(p + "attn_output", loader.get(p + "attn_output.weight"))
        self.attn = dna_semantic_compression.GenomicAttention(n_head, n_head_kv, head_dim, rmsnorm_weight.tolist() if rmsnorm_weight is not None else [], eps)

    def forward(self, x, pos):
        x_norm = self.attn.apply_rmsnorm(x.tolist())
        q, k, v = self.q_gen.forward(np.array(x_norm)), self.k_gen.forward(np.array(x_norm)), self.v_gen.forward(np.array(x_norm))
        return self.w_o.forward(np.array(self.attn.forward_attention(q.tolist(), k.tolist(), v.tolist(), pos)))

class GenomicTransformerBlock:
    def __init__(self, loader, idx, n_head, n_head_kv, head_dim, rope_base, eps):
        p = f"blk.{idx}."
        self.attn = GenomicAttentionLayer(loader, p, n_head, n_head_kv, head_dim, rope_base, rmsnorm_weight=loader.get(p + "attn_norm.weight").data.astype(np.float32), eps=eps)
        g, u = loader.get(p + "ffn_gate.weight"), loader.get(p + "ffn_up.weight")
        self.gate_gen, self.up_gen = GenomicLayer(p + "gate", g), GenomicLayer(p + "up", u)
        self.w_down = GenomicLayer(p + "ffn_down", loader.get(p + "ffn_down.weight"))
        self.ffn_norm, self.eps = loader.get(p + "ffn_norm.weight").data.astype(np.float32), eps

    def forward(self, x, pos):
        x = x + self.attn.forward(x, pos)
        rms = np.sqrt(np.mean(x**2) + self.eps)
        x_norm = (x / rms) * self.ffn_norm
        gate = self.gate_gen.forward(x_norm)
        up = self.up_gen.forward(x_norm)
        # SiLU (Swish)
        swiglu_out = (gate / (1.0 + np.exp(-gate))) * up
        return x + self.w_down.forward(swiglu_out)

class GenomicLLM:
    def __init__(self, model_path, num_blocks=None):
        start_total = time.time()
        self.reader = gguf.GGUFReader(model_path)
        loader = TensorLoader(self.reader)
        
        arch = self.reader.fields["general.architecture"].parts[-1]
        if hasattr(arch, 'tolist'): arch = arch.tolist()
        arch = ("".join([chr(x) for x in arch]) if isinstance(arch, list) and isinstance(arch[0], int) else str(arch[0] if isinstance(arch, list) else arch)).strip().replace("\x00", "")
        self.n_embd, self.n_head, self.n_head_kv = int(self.reader.fields[f"{arch}.embedding_length"].parts[-1][0]), int(self.reader.fields[f"{arch}.attention.head_count"].parts[-1][0]), int(self.reader.fields[f"{arch}.attention.head_count_kv"].parts[-1][0])
        self.head_dim, self.n_blocks, self.eps = self.n_embd // self.n_head, int(self.reader.fields[f"{arch}.block_count"].parts[-1][0]) if num_blocks is None else num_blocks, float(self.reader.fields[f"{arch}.attention.layer_norm_rms_epsilon"].parts[-1][0])
        rope_base_field = f"{arch}.rope.freq_base"
        self.rope_base = float(self.reader.fields[rope_base_field].parts[-1][0]) if rope_base_field in self.reader.fields else 10000.0
        
        # Forzamos anclas totales (-1.0 threshold) para las capas más frágiles
        embeddings = GenomicLayer("token_embd", loader.get("token_embd.weight"), balancer=None, anchor_threshold=-1.0)
        output_norm = loader.get("output_norm.weight").data.astype(np.float32).tolist()
        
        w_head = loader.get("output.weight", required=False)
        if w_head is None:
             w_head = loader.get("token_embd.weight")
             
        lm_head = GenomicLayer("lm_head", w_head, balancer=None, anchor_threshold=-1.0)
        
        rust_blocks = []
        for i in range(self.n_blocks):
            p = f"blk.{i}."
            # Protegemos Q, K, V, O con anclas totales (-1.0) para asegurar coherencia en la atención
            q_gen = GenomicLayer(p + "attn_q", loader.get(p + "attn_q.weight"), n_head=self.n_head, head_dim=self.head_dim, balancer=None, anchor_threshold=-1.0)
            k_gen = GenomicLayer(p + "attn_k", loader.get(p + "attn_k.weight"), n_head=self.n_head_kv, head_dim=self.head_dim, balancer=None, anchor_threshold=-1.0)
            v_gen = GenomicLayer(p + "attn_v", loader.get(p + "attn_v.weight"), balancer=None, anchor_threshold=-1.0)
            w_o = GenomicLayer(p + "attn_output", loader.get(p + "attn_output.weight"), balancer=None, anchor_threshold=-1.0)
            attn_norm = loader.get(p + "attn_norm.weight").data.astype(np.float32).tolist()
            attn = dna_semantic_compression.GenomicAttention(self.n_head, self.n_head_kv, self.head_dim, attn_norm, self.eps, self.rope_base)
            
            # Las capas FFN siguen en 2-bit, pero con mayor densidad de anclas (threshold 0.75) para evitar degradación de señal
            g, u = loader.get(p + "ffn_gate.weight"), loader.get(p + "ffn_up.weight")
            gate_gen, up_gen = GenomicLayer(p + "gate", g, balancer=None, anchor_threshold=0.75), GenomicLayer(p + "up", u, balancer=None, anchor_threshold=0.75)
            w_down = GenomicLayer(p + "ffn_down", loader.get(p + "ffn_down.weight"), balancer=None, anchor_threshold=0.75)
            ffn_norm = loader.get(p + "ffn_norm.weight").data.astype(np.float32).tolist()
            
            rust_block = dna_semantic_compression.RustGenomicBlock(
                i, attn, q_gen.linear, k_gen.linear, v_gen.linear, w_o.linear, 
                gate_gen.linear, up_gen.linear, w_down.linear, ffn_norm, self.eps
            )
            rust_blocks.append(rust_block)
            
        self.rust_llm = dna_semantic_compression.RustGenomicLLM(
            embeddings.linear, rust_blocks, output_norm, lm_head.linear, self.eps
        )
        
        end_total = time.time()
        print(f"[*] Evolución 4: Sincronización Genómica Nativa finalizada en {end_total - start_total:.2f}s")
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B" if arch == "qwen2" else "HuggingFaceTB/SmolLM2-135M-Instruct")

    def forward(self, tokens, clear_cache=True):
        all_logits = []
        for tid in (tokens if isinstance(tokens, list) else [tokens]):
            # Entire forward pass now happens purely in Rust SIMD per token
            logits = self.rust_llm.forward(tid, clear_cache)
            all_logits.append(logits)
            clear_cache = False # Only clear on first token of prompt
        return np.stack(all_logits)

    def generate(self, prompt, max_new_tokens=20, temperature=0.7, top_p=0.9, repetition_penalty=1.2):
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        generated_tokens = tokens.copy()
        
        # Inferencia inicial (prompt)
        next_token_logits = self.forward(tokens, clear_cache=True)[-1]
        
        for _ in range(max_new_tokens):
            # Aplicar penalización de repetición nativa en Rust
            penalized_logits = dna_semantic_compression.apply_repetition_penalty(
                next_token_logits.tolist(), 
                repetition_penalty, 
                generated_tokens[-20:] # Mirar últimos 20 tokens
            )
            
            # Muestreo Top-P nativo en Rust
            next_id = dna_semantic_compression.sample_top_p(
                penalized_logits, 
                temperature, 
                top_p
            )
            
            if next_id == self.tokenizer.eos_token_id: 
                break
                
            generated_tokens.append(next_id)
            yield self.tokenizer.decode([next_id])
            
            # Siguiente paso de inferencia (incremental)
            next_token_logits = self.forward([next_id], clear_cache=False)[-1]
