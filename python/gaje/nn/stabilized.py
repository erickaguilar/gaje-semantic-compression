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
        self.balancer = balancer or SignalToNoiseBalancer()
        is_q_or_k = "attn_q" in name or "attn_k" in name
        
        if hasattr(weights_f32_or_tensor, 'tensor_type'):
            tensor = weights_f32_or_tensor
            self.out_features, self.in_features = tensor.shape[::-1] if len(tensor.shape) == 2 else (tensor.shape[0], 1)
            if tensor.tensor_type in [gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32]:
                weights_f32 = np.array(tensor.data, dtype=np.float32)
                if is_q_or_k and n_head is not None and head_dim is not None:
                    w_matrix = weights_f32.reshape(self.out_features, self.in_features)
                    w_new = np.zeros_like(w_matrix)
                    actual_heads = min(n_head, self.out_features // head_dim)
                    for h in range(actual_heads):
                        for i in range(head_dim // 2):
                            if h * head_dim + head_dim // 2 + i < self.out_features:
                                w_new[h * head_dim + i] = w_matrix[h * head_dim + 2 * i]
                                w_new[h * head_dim + head_dim // 2 + i] = w_matrix[h * head_dim + 2 * i + 1]
                    weights_f32 = w_new.flatten()
                self._init_from_f32(weights_f32, block_size, anchor_threshold)
            else:
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
        print(f"    [+] Layer {name}: {self.out_features}x{self.in_features}, DB: {len(self.dna_database)} bytes")

    def _init_from_f32(self, weights_f32, block_size, anchor_threshold):
        reshaped = weights_f32.reshape(-1, block_size)
        base_centroids = np.array([-1.510, -0.4528, 0.4528, 1.510], dtype=np.float32)
        w_matrix = weights_f32.reshape(self.out_features, self.in_features)
        entropies = np.array(dna_semantic_compression.calculate_shannon_entropy(w_matrix.tolist()))
        full_mask = self.balancer.generate_precision_mask(entropies)
        self.precision_mask = [int(np.max(full_mask[i:i+4])) for i in range(0, self.in_features, 4)]
        dna_batch, epi_batch, tri_batch = [], [], []
        all_dna_centroids, all_epi_centroids, all_tri_centroids = [], [], []
        anchors_f32 = np.zeros_like(weights_f32).reshape(-1)
        for i in range(len(reshaped)):
            block_idx_in_row = i % (self.in_features // block_size)
            block = reshaped[i]
            block_mean, block_std = np.mean(block), np.std(block) + 1e-6
            t_base = [block_mean - 1.0 * block_std, block_mean, block_mean + 1.0 * block_std]
            dna = dna_semantic_compression.quantize_embedding(block.tolist(), t_base)
            dna_batch.append(dna)
            c_base = (block_mean + base_centroids * block_std).astype(np.float32)
            all_dna_centroids.extend(c_base.tolist())
            residual = block - np.array(dna_semantic_compression.dequantize_embedding(dna, block_size, c_base.tolist()))
            mode = np.max(full_mask[block_idx_in_row * block_size : (block_idx_in_row + 1) * block_size])
            if mode >= 1:
                e_mean, e_std = np.mean(residual), np.std(residual) + 1e-6
                epi = dna_semantic_compression.quantize_embedding(residual.tolist(), [e_mean - 1.0 * e_std, e_mean, e_mean + 1.0 * e_std])
                epi_batch.append(epi)
                c_epi = (e_mean + base_centroids * e_std).astype(np.float32)
                all_epi_centroids.extend(c_epi.tolist())
                residual = residual - np.array(dna_semantic_compression.dequantize_embedding(epi, block_size, c_epi.tolist()))
            else:
                epi_batch.append(b"\x00" * (block_size // 4))
                all_epi_centroids.extend([0.0] * 4)
            if mode >= 2:
                t_mean, t_std = np.mean(residual), np.std(residual) + 1e-6
                tri = dna_semantic_compression.quantize_embedding(residual.tolist(), [t_mean - 1.0 * t_std, t_mean, t_mean + 1.0 * t_std])
                tri_batch.append(tri)
                c_tri = (t_mean + base_centroids * t_std).astype(np.float32)
                all_tri_centroids.extend(c_tri.tolist())
                residual = residual - np.array(dna_semantic_compression.dequantize_embedding(tri, block_size, c_tri.tolist()))
            else:
                tri_batch.append(b"\x00" * (block_size // 4))
                all_tri_centroids.extend([0.0] * 4)
            err_threshold = np.quantile(np.abs(residual), anchor_threshold)
            anchor_mask = np.abs(residual) >= err_threshold
            anchors_f32[i * block_size : (i + 1) * block_size][anchor_mask] = residual[anchor_mask]
        self.dna_database, self.epigenetic_database, self.triplet_database = b"".join(dna_batch), b"".join(epi_batch), b"".join(tri_batch)
        self.dna_centroids, self.epigenetic_centroids, self.triplet_centroids = all_dna_centroids, all_epi_centroids, all_tri_centroids
        self.anchors = anchors_f32.tolist()
        stride = block_size // 4
        self.precision_mask = [int(np.max(full_mask[b*block_size+k*4 : b*block_size+(k+1)*4])) for b in range(self.in_features // block_size) for k in range(stride)] * self.out_features

    def forward(self, x):
        return np.array(self.linear.forward(x.tolist()), dtype=np.float32)
        
    def get_row(self, idx):
        n_blocks, stride = self.in_features // self.block_size, self.block_size // 4
        dna_row = self.linear.database[idx * n_blocks * stride : (idx + 1) * n_blocks * stride]
        res = np.zeros(self.in_features, dtype=np.float32)
        for b in range(n_blocks):
            res[b*self.block_size : (b+1)*self.block_size] = np.array(dna_semantic_compression.dequantize_embedding(dna_row[b*stride : (b+1)*stride], self.block_size, self.linear.centroids[(idx * n_blocks + b) * 4 : (idx * n_blocks + b) * 4 + 4]))
        return res + np.array(self.linear.anchors[idx * self.in_features : (idx + 1) * self.in_features])

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
        gate_gen, up_gen = GenomicLayer(p + "gate", g), GenomicLayer(p + "up", u)
        self.swiglu = dna_semantic_compression.GenomicSwiGLU(gate_gen.dna_database, up_gen.dna_database, gate_gen.dna_centroids, gate_gen.out_features, gate_gen.in_features, gate_gen.block_size)
        self.w_down = GenomicLayer(p + "ffn_down", loader.get(p + "ffn_down.weight"))
        self.ffn_norm, self.eps = loader.get(p + "ffn_norm.weight").data.astype(np.float32), eps

    def forward(self, x, pos):
        x = x + self.attn.forward(x, pos)
        rms = np.sqrt(np.mean(x**2) + self.eps)
        return x + self.w_down.forward(np.array(self.swiglu.forward(((x / rms) * self.ffn_norm).tolist())))

class GenomicLLM:
    def __init__(self, model_path, num_blocks=None):
        print(f"🧬 Sincronizando Organismo Genómico (2-bit): {os.path.basename(model_path)}")
        self.reader = gguf.GGUFReader(model_path)
        loader = TensorLoader(self.reader)
        
        arch = self.reader.fields["general.architecture"].parts[-1]
        if hasattr(arch, 'tolist'): arch = arch.tolist()
        arch = ("".join([chr(x) for x in arch]) if isinstance(arch, list) and isinstance(arch[0], int) else str(arch[0] if isinstance(arch, list) else arch)).strip().replace("\x00", "")
        self.n_embd, self.n_head, self.n_head_kv = int(self.reader.fields[f"{arch}.embedding_length"].parts[-1][0]), int(self.reader.fields[f"{arch}.attention.head_count"].parts[-1][0]), int(self.reader.fields[f"{arch}.attention.head_count_kv"].parts[-1][0])
        self.head_dim, self.n_blocks, self.eps = self.n_embd // self.n_head, int(self.reader.fields[f"{arch}.block_count"].parts[-1][0]) if num_blocks is None else num_blocks, float(self.reader.fields[f"{arch}.attention.layer_norm_rms_epsilon"].parts[-1][0])
        
        self.embeddings = GenomicLayer("token_embd", loader.get("token_embd.weight"), balancer=SignalToNoiseBalancer())
        self.output_norm = loader.get("output_norm.weight").data.astype(np.float32)
        
        w_head = loader.get("output.weight", required=False)
        if w_head is None:
             w_head = loader.get("token_embd.weight")
             
        self.lm_head = GenomicLayer("lm_head", w_head, balancer=SignalToNoiseBalancer())
        self.blocks = [GenomicTransformerBlock(loader, i, self.n_head, self.n_head_kv, self.head_dim, 0, self.eps) for i in range(self.n_blocks)]
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B" if arch == "qwen2" else "HuggingFaceTB/SmolLM2-135M-Instruct")

    def forward(self, tokens, clear_cache=True):
        if clear_cache:
            for b in self.blocks: b.attn.attn.clear_cache()
        all_logits = []
        for tid in (tokens if isinstance(tokens, list) else [tokens]):
            h, pos = self.embeddings.get_row(tid), self.blocks[0].attn.attn.k_cache_len
            for block in self.blocks: h = block.forward(h, pos)
            rms = np.sqrt(np.mean(h**2) + self.eps)
            all_logits.append(self.lm_head.forward((h / rms) * self.output_norm))
        return np.stack(all_logits)

    def generate(self, prompt, max_new_tokens=20, temperature=0.7):
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        next_token_logits = self.forward(tokens, clear_cache=True)[-1]
        for _ in range(max_new_tokens):
            next_id = int(np.argmax(next_token_logits / max(temperature, 1e-6)))
            if next_id == self.tokenizer.eos_token_id: break
            next_token_logits = self.forward([next_id], clear_cache=False)[-1]
            yield self.tokenizer.decode([next_id])
