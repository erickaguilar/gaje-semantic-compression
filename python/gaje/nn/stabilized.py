import os
import numpy as np
from gaje.core import _impl as dna_semantic_compression
import gguf
import time
from transformers import AutoTokenizer

from gaje.utils.quantization import dequantize_q8_0
from gaje.processing.balancer import SignalToNoiseBalancer

class GenomicLayer:
    def __init__(self, name, weights_f32_or_tensor, block_size=32, anchor_threshold=0.98, rmsnorm_weight=None, eps=1e-6, balancer=None, n_head=None, head_dim=None):
        self.name = name
        self.block_size = block_size
        self.balancer = balancer or SignalToNoiseBalancer()
        is_q_or_k = "attn_q" in name or "attn_k" in name
        
        # 1. DIRECT GENOMIC INGESTION (DGI) - Bypass Q8 loss
        if hasattr(weights_f32_or_tensor, 'tensor_type'):
            tensor = weights_f32_or_tensor
            self.out_features, self.in_features = tensor.shape[::-1] if len(tensor.shape) == 2 else (tensor.shape[0], 1)
            
            # Use balancer to decide initial threshold
            a_threshold = self.balancer.current_threshold
            
            if tensor.tensor_type in [gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32]:
                # De-permute F16/F32 if needed before genomizing
                weights_f32 = np.array(tensor.data, dtype=np.float32)
                if is_q_or_k and n_head is not None and head_dim is not None:
                    # GGUF weights are usually (out, in). Permutation applies to the 'out' (heads) dimension.
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
                # Fallback to standard dequantization (Q8_0, Q4_K, etc.)
                weights_f32 = dequantize_q8_0(tensor, n_head, head_dim, is_q_or_k)
                self._init_from_f32(weights_f32, block_size, anchor_threshold)
        else:
            weights_f32 = weights_f32_or_tensor
            self.out_features, self.in_features = weights_f32.shape
            self._init_from_f32(weights_f32, block_size, anchor_threshold)

        print(f"    [+] Layer {name}: {self.out_features}x{self.in_features}, DB: {len(self.dna_database)} bytes")
        self.linear = dna_semantic_compression.GenomicLinear(
            self.dna_database,
            self.anchors,
            self.dna_centroids,
            self.out_features,
            self.in_features,
            self.block_size,
            rmsnorm_weight.tolist() if rmsnorm_weight is not None else [],
            eps,
            self.precision_mask,
            self.epigenetic_database,
            self.epigenetic_centroids,
            self.triplet_database,
            self.triplet_centroids
        )

    def _init_from_f32(self, weights_f32, block_size, anchor_threshold):
        reshaped = weights_f32.reshape(-1, block_size)
        base_centroids = np.array([-1.510, -0.4528, 0.4528, 1.510], dtype=np.float32)
        
        # Phase 12: Entropy Mapping
        # Calculate entropy per dimension across all output features
        w_matrix = weights_f32.reshape(self.out_features, self.in_features)
        entropies = np.array(dna_semantic_compression.calculate_shannon_entropy(w_matrix.tolist()))
        
        # Generate mask per dimension
        full_mask = self.balancer.generate_precision_mask(entropies)
        
        # Downsample mask to per-byte (4 dims) as expected by Rust kernel
        # We take the maximum mode in each 4-dim group to be safe
        self.precision_mask = []
        for i in range(0, self.in_features, 4):
            group = full_mask[i:i+4]
            self.precision_mask.append(int(np.max(group) if len(group) > 0 else 0))
            
        dna_batch = []
        epi_batch = []
        tri_batch = []
        all_dna_centroids = []
        all_epi_centroids = []
        all_tri_centroids = []
        
        anchors_f32 = np.zeros_like(weights_f32).reshape(-1)
        
        for i in range(len(reshaped)):
            # Find the mode for this block by looking at the mask
            # Actually, the Rust kernel expects a mask that is indexed by [j * stride + k]
            # where j is block index and k is byte index.
            # But here we have a mask that is per-dimension-index.
            # If the mask is SHARED for all rows, then we need to repeat it.
            # However, the Rust code uses: self.precision_mask[j * self.stride + k]
            # This implies the mask is for the WHOLE database (out_features * in_features / 4).
            
            block_idx_in_row = i % (self.in_features // block_size)
            block = reshaped[i]
            
            # 1. Base Layer (2-bit)
            block_mean = np.mean(block)
            block_std = np.std(block) + 1e-6
            t_base = [block_mean - 1.0 * block_std, block_mean, block_mean + 1.0 * block_std]
            dna = dna_semantic_compression.quantize_embedding(block.tolist(), t_base)
            dna_batch.append(dna)
            
            c_base = (block_mean + base_centroids * block_std).astype(np.float32)
            all_dna_centroids.extend(c_base.tolist())
            
            dequant_base = np.array(dna_semantic_compression.dequantize_embedding(dna, block_size, c_base.tolist()))
            residual = block - dequant_base
            
            # 2. Epigenetic Layer (4-bit)
            # Check if any dimension in this block requires higher precision
            block_mask = full_mask[block_idx_in_row * block_size : (block_idx_in_row + 1) * block_size]
            mode = np.max(block_mask)
            
            if mode >= 1:
                e_mean = np.mean(residual)
                e_std = np.std(residual) + 1e-6
                t_epi = [e_mean - 1.0 * e_std, e_mean, e_mean + 1.0 * e_std]
                epi = dna_semantic_compression.quantize_embedding(residual.tolist(), t_epi)
                epi_batch.append(epi)
                c_epi = (e_mean + base_centroids * e_std).astype(np.float32)
                all_epi_centroids.extend(c_epi.tolist())
                dequant_epi = np.array(dna_semantic_compression.dequantize_embedding(epi, block_size, c_epi.tolist()))
                residual = residual - dequant_epi
            else:
                epi_batch.append(b"\x00" * (block_size // 4))
                all_epi_centroids.extend([0.0] * 4)
                
            # 3. Triplet Layer (6-bit)
            if mode >= 2:
                t_mean = np.mean(residual)
                t_std = np.std(residual) + 1e-6
                t_tri = [t_mean - 1.0 * t_std, t_mean, t_mean + 1.0 * t_std]
                tri = dna_semantic_compression.quantize_embedding(residual.tolist(), t_tri)
                tri_batch.append(tri)
                c_tri = (t_mean + base_centroids * t_std).astype(np.float32)
                all_tri_centroids.extend(c_tri.tolist())
                dequant_tri = np.array(dna_semantic_compression.dequantize_embedding(tri, block_size, c_tri.tolist()))
                residual = residual - dequant_tri
            else:
                tri_batch.append(b"\x00" * (block_size // 4))
                all_tri_centroids.extend([0.0] * 4)

            # Anchors on the final residual
            err_threshold = np.quantile(np.abs(residual), anchor_threshold)
            anchor_mask = np.abs(residual) >= err_threshold
            offset = i * block_size
            anchors_f32[offset : offset + block_size][anchor_mask] = residual[anchor_mask]

        self.dna_database = b"".join(dna_batch)
        self.epigenetic_database = b"".join(epi_batch)
        self.triplet_database = b"".join(tri_batch)
        self.dna_centroids = all_dna_centroids
        self.epigenetic_centroids = all_epi_centroids
        self.triplet_centroids = all_tri_centroids
        self.anchors = anchors_f32.tolist()
        
        # Finally, the precision_mask passed to Rust should be for the WHOLE database
        # as indexed by [j * stride + k] in the kernel.
        # Since the mask is shared per dimension index, we tile it.
        stride = block_size // 4
        n_blocks_per_row = self.in_features // block_size
        row_mask = []
        for b in range(n_blocks_per_row):
            block_m = full_mask[b*block_size : (b+1)*block_size]
            for k in range(stride):
                row_mask.append(int(np.max(block_m[k*4 : (k+1)*4])))
        
        self.precision_mask = row_mask * self.out_features

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
        q_gen = GenomicLayer(p + "attn_q", t_q, n_head=n_head, head_dim=head_dim)
        k_gen = GenomicLayer(p + "attn_k", t_k, n_head=n_head_kv, head_dim=head_dim)
        v_gen = GenomicLayer(p + "attn_v", t_v)
        
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
            head_dim,
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
        # Detect architecture
        if "general.architecture" in self.reader.fields:
            part = self.reader.fields["general.architecture"].parts[-1]
            if hasattr(part, 'tolist'): part = part.tolist()
            if isinstance(part, list):
                if isinstance(part[0], int):
                    arch = "".join([chr(x) for x in part])
                else:
                    arch = part[0]
                    if isinstance(arch, (bytes, bytearray)): arch = arch.decode("utf-8")
            else:
                arch = str(part)
        else:
            arch = "llama"
        
        # Clean up arch string (remove nulls or extra chars)
        arch = arch.strip().replace("\x00", "")
        print(f"[*] Arquitectura detectada: '{arch}'")

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
            
        tokenizer_name = "Qwen/Qwen2-0.5B" if arch == "qwen2" else "HuggingFaceTB/SmolLM2-135M-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

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
            
        from gaje.core.archive import GAJEArchive
        
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
            
        print(f"✅ Modelo guardado exitosamente.")
