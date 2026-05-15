import os
import numpy as np
import gaje.core._impl as dna_semantic_compression
import gguf
import time
from transformers import AutoTokenizer

from gaje.utils.quantization import dequantize_q8_0, unpermute_llama_weights
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
    def __init__(self, name, weights_f32_or_tensor, bias_f32_or_tensor=None, block_size=32, anchor_threshold=-1.0, rmsnorm_weight=None, eps=1e-6, balancer=None, n_head=None, head_dim=None, config=None):
        self.name = name
        self.block_size = block_size
        self.balancer = balancer
        self.config = config
        is_q_or_k = "attn_q" in name or "attn_k" in name
        
        if hasattr(weights_f32_or_tensor, 'tensor_type'):
            tensor = weights_f32_or_tensor
            
            # GGUF tensors are usually [in_features, out_features] or similar
            if len(tensor.shape) == 2:
                self.in_features, self.out_features = tensor.shape
            else:
                self.out_features, self.in_features = tensor.shape[0], 1
            
            if tensor.tensor_type in [gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32]:
                if tensor.tensor_type == gguf.GGMLQuantizationType.F16:
                    raw_data = np.frombuffer(tensor.data, dtype=np.float16).astype(np.float32)
                else:
                    raw_data = np.frombuffer(tensor.data, dtype=np.float32)
                
                # GGUF: first dim is fastest. Correct reshape is [out, in]
                w_matrix = raw_data.reshape(self.out_features, self.in_features)
                
                # Para RoPE Interleaved en Rust, necesitamos des-permutar de Split a Interleaved
                unpermute = self.config.unpermute_weights if self.config else True
                if unpermute and is_q_or_k and n_head is not None and head_dim is not None:
                    from gaje.utils.quantization import unpermute_to_interleaved
                    w_matrix = unpermute_to_interleaved(w_matrix, n_head, head_dim)
                
                self._init_from_f32(w_matrix, block_size, anchor_threshold)
            else:
                # Solo para Q8_0 aplicamos la lógica de des-permutación necesaria
                weights_f32 = dequantize_q8_0(tensor, n_head, head_dim, is_q_or_k)
                self.out_features, self.in_features = weights_f32.shape
                self._init_from_f32(weights_f32, block_size, anchor_threshold)
        else:
            weights_f32 = weights_f32_or_tensor
            self.out_features, self.in_features = weights_f32.shape
            
            # También para pesos brutos (born-genomic o manual), respetamos el flag
            unpermute = self.config.unpermute_weights if self.config else False # Default false for raw
            if unpermute and is_q_or_k and n_head is not None and head_dim is not None:
                from gaje.utils.quantization import unpermute_to_interleaved
                weights_f32 = unpermute_to_interleaved(weights_f32, n_head, head_dim)
                
            self._init_from_f32(weights_f32, block_size, anchor_threshold)

        # Manejo de Bias (Crucial para Qwen2)
        self.bias = []
        if bias_f32_or_tensor is not None:
            if hasattr(bias_f32_or_tensor, 'data'):
                b_data = bias_f32_or_tensor.data
                if bias_f32_or_tensor.tensor_type == gguf.GGMLQuantizationType.F16:
                    self.bias = np.frombuffer(b_data, dtype=np.float16).astype(np.float32).tolist()
                else:
                    self.bias = np.frombuffer(b_data, dtype=np.float32).tolist()
            else:
                self.bias = bias_f32_or_tensor.tolist() if hasattr(bias_f32_or_tensor, "tolist") else list(bias_f32_or_tensor)

        self.linear = dna_semantic_compression.GenomicLinear(
            self.dna_database, self.anchors, self.dna_centroids, self.out_features, self.in_features, self.block_size,
            rmsnorm_weight=rmsnorm_weight.tolist() if rmsnorm_weight is not None else [], 
            eps=eps, 
            precision_mask=self.precision_mask,
            epigenetic_database=self.epigenetic_database, 
            epigenetic_centroids=self.epigenetic_centroids, 
            triplet_database=self.triplet_database, 
            triplet_centroids=self.triplet_centroids,
            bias=self.bias
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
        return np.array(self.linear.forward(x.tolist() if hasattr(x, "tolist") else x), dtype=np.float32)
        
    def refine_centroids(self, x, target, lr=0.01):
        """Refines the centroids of this layer based on input and target output."""
        self.linear.refine_centroids(x.tolist() if hasattr(x, "tolist") else x, 
                                     target.tolist() if hasattr(target, "tolist") else target, 
                                     lr)

    def refine_with_grads(self, x, grads, lr=0.01):
        """Refines the centroids of this layer based on input and gradients."""
        self.linear.refine_with_grads(x.tolist() if hasattr(x, "tolist") else x, 
                                      grads.tolist() if hasattr(grads, "tolist") else grads, 
                                      lr)

    def get_row(self, idx):
        n_blocks, stride = self.in_features // self.block_size, self.block_size // 4
        dna_row = self.dna_database[idx * n_blocks * stride : (idx + 1) * n_blocks * stride]
        res = np.zeros(self.in_features, dtype=np.float32)
        for b in range(n_blocks):
            res[b*self.block_size : (b+1)*self.block_size] = np.array(dna_semantic_compression.dequantize_embedding(dna_row[b*stride : (b+1)*stride], self.block_size, self.dna_centroids[(idx * n_blocks + b) * 4 : (idx * n_blocks + b) * 4 + 4]))
        return res + np.array(self.anchors[idx * self.in_features : (idx + 1) * self.in_features], dtype=np.float32)

class GenomicAttentionLayer:
    def __init__(self, loader, p, n_head, n_head_kv, head_dim, rope_base, rmsnorm_weight=None, eps=1e-6, config=None):
        self.config = config
        self.q_gen = GenomicLayer(p + "attn_q", loader.get(p + "attn_q.weight"), bias_f32_or_tensor=loader.get(p + "attn_q.bias", required=False), n_head=n_head, head_dim=head_dim, balancer=None, anchor_threshold=-1.0, config=config)
        self.k_gen = GenomicLayer(p + "attn_k", loader.get(p + "attn_k.weight"), bias_f32_or_tensor=loader.get(p + "attn_k.bias", required=False), n_head=n_head_kv, head_dim=head_dim, balancer=None, anchor_threshold=-1.0, config=config)
        self.v_gen = GenomicLayer(p + "attn_v", loader.get(p + "attn_v.weight"), bias_f32_or_tensor=loader.get(p + "attn_v.bias", required=False), balancer=None, anchor_threshold=-1.0, config=config)
        self.w_o = GenomicLayer(p + "attn_output", loader.get(p + "attn_output.weight"), bias_f32_or_tensor=loader.get(p + "attn_output.bias", required=False), balancer=None, anchor_threshold=-1.0, config=config)
        self.attn = dna_semantic_compression.GenomicAttention(n_head, n_head_kv, head_dim, rmsnorm_weight.tolist() if rmsnorm_weight is not None else [], eps, rope_base)

    def forward(self, x, pos):
        x_norm = self.attn.apply_rmsnorm(x.tolist() if hasattr(x, "tolist") else x)
        q, k, v = self.q_gen.forward(np.array(x_norm)), self.k_gen.forward(np.array(x_norm)), self.v_gen.forward(np.array(x_norm))
        return self.w_o.forward(np.array(self.attn.forward_attention(q.tolist(), k.tolist(), v.tolist(), pos)))

class GenomicTransformerBlock:
    def __init__(self, loader, idx, n_head, n_head_kv, head_dim, rope_base, eps, anchor_threshold=0.1, config=None):
        self.config = config
        p = f"blk.{idx}."
        attn_norm_data = loader.get(p + "attn_norm.weight").data.astype(np.float32)
        self.attn_layer = GenomicAttentionLayer(loader, p, n_head, n_head_kv, head_dim, rope_base, rmsnorm_weight=attn_norm_data, eps=eps, config=config)
        g, u = loader.get(p + "ffn_gate.weight"), loader.get(p + "ffn_up.weight")
        self.gate_gen, self.up_gen = GenomicLayer(p + "gate", g, balancer=None, anchor_threshold=anchor_threshold, config=config), GenomicLayer(p + "up", u, balancer=None, anchor_threshold=anchor_threshold, config=config)
        self.w_down = GenomicLayer(p + "ffn_down", loader.get(p + "ffn_down.weight"), balancer=None, anchor_threshold=anchor_threshold, config=config)
        self.ffn_norm = loader.get(p + "ffn_norm.weight").data.astype(np.float32).tolist()
        self.eps = eps
        
        # Reference to the Rust block
        self.rust_block = dna_semantic_compression.RustGenomicBlock(
            idx, self.attn_layer.attn, self.attn_layer.q_gen.linear, self.attn_layer.k_gen.linear, 
            self.attn_layer.v_gen.linear, self.attn_layer.w_o.linear, 
            self.gate_gen.linear, self.up_gen.linear, self.w_down.linear, self.ffn_norm, self.eps
        )

    def forward(self, x, pos):
        return np.array(self.rust_block.forward(x.tolist() if hasattr(x, "tolist") else x, pos), dtype=np.float32)

    def refine_swiglu(self, x_norm, target, lr=0.01):
        """Native SwiGLU refinement."""
        self.rust_block.refine_swiglu(x_norm.tolist() if hasattr(x_norm, "tolist") else x_norm, 
                                      target.tolist() if hasattr(target, "tolist") else target, 
                                      lr)

    def refine_attention(self, x_norm, target, pos, lr=0.01):
        """Native attention projection refinement."""
        self.rust_block.refine_attention(x_norm.tolist() if hasattr(x_norm, "tolist") else x_norm, 
                                         target.tolist() if hasattr(target, "tolist") else target, 
                                         pos, lr)

from gaje.nn.configs import get_config, detect_arch, ArchitectureConfig, ARCHITECTURES

class GenomicLLM:
    def __init__(self, model_path=None, num_blocks=None, config=None):
        start_total = time.time()
        
        if model_path:
            self.reader = gguf.GGUFReader(model_path)
            loader = TensorLoader(self.reader)
            arch_name = detect_arch(self.reader)
            self.config = config or get_config(arch_name)
            print(f"[*] Arquitectura detectada: {arch_name} (Usando config: {self.config.name})")
            
            self.n_embd = int(self.reader.fields[f"{arch_name}.embedding_length"].parts[-1][0])
            self.n_head = int(self.reader.fields[f"{arch_name}.attention.head_count"].parts[-1][0])
            self.n_head_kv = int(self.reader.fields[f"{arch_name}.attention.head_count_kv"].parts[-1][0])
            self.head_dim = self.n_embd // self.n_head
            self.n_blocks = int(self.reader.fields[f"{arch_name}.block_count"].parts[-1][0]) if num_blocks is None else num_blocks
            self.eps = float(self.reader.fields[f"{arch_name}.attention.layer_norm_rms_epsilon"].parts[-1][0])
            
            rope_base_field = f"{arch_name}.rope.freq_base"
            if rope_base_field in self.reader.fields:
                self.rope_base = float(self.reader.fields[rope_base_field].parts[-1][0])
            else:
                self.rope_base = self.config.rope_base
                
            if self.config.apply_smollm_rope_patch and self.rope_base > 1000000.0:
                print(f"[!] Aviso: Forzando RoPE Base a 10000.0 para {os.path.basename(model_path)}")
                self.rope_base = 10000.0
        else:
            # Born-genomic initialization
            if config is None:
                raise ValueError("Must provide config for born-genomic initialization")
            self.config = config
            self.n_embd = 768 
            self.n_head = 12
            self.n_head_kv = 12
            self.head_dim = self.n_embd // self.n_head
            self.n_blocks = num_blocks or 12
            self.eps = 1e-6
            self.rope_base = self.config.rope_base
            loader = None
            print(f"🧬 Iniciando Organismo GAJE Nativo (Born-Genomic): {self.config.name}")

        print(f"[*] RoPE Base: {self.rope_base}")
        
        # Initialization logic
        if loader:
            embeddings = GenomicLayer("token_embd", loader.get("token_embd.weight"), balancer=None, anchor_threshold=-1.0, config=self.config)
            output_norm = loader.get("output_norm.weight").data.astype(np.float32).tolist()
            w_head = loader.get("output.weight", required=False) or loader.get("token_embd.weight")
            lm_head = GenomicLayer("lm_head", w_head, balancer=None, anchor_threshold=0.1, config=self.config)
        else:
            emb_w = np.random.normal(0, 0.02, (50257, self.n_embd)).astype(np.float32) 
            embeddings = GenomicLayer("token_embd", emb_w, balancer=None, anchor_threshold=-1.0, config=self.config)
            output_norm = np.ones(self.n_embd).astype(np.float32).tolist()
            lm_head_w = np.random.normal(0, 0.02, (50257, self.n_embd)).astype(np.float32)
            lm_head = GenomicLayer("lm_head", lm_head_w, balancer=None, anchor_threshold=0.1, config=self.config)
        
        rust_blocks = []
        self.blocks = []
        for i in range(self.n_blocks):
            if loader:
                block = GenomicTransformerBlock(loader, i, self.n_head, self.n_head_kv, self.head_dim, self.rope_base, self.eps, anchor_threshold=0.1, config=self.config)
            else:
                block = self._create_random_block(i)
                
            self.blocks.append(block)
            rust_blocks.append(block.rust_block)
            if (i+1) % 10 == 0: print(f"    [~] Bloque {i+1}/{self.n_blocks} sincronizado...")
            
        self.rust_llm = dna_semantic_compression.RustGenomicLLM(
            embeddings.linear, rust_blocks, output_norm, lm_head.linear, self.eps
        )
        
        end_total = time.time()
        print(f"[*] Sincronización Genómica Nativa finalizada en {end_total - start_total:.2f}s")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_id)

    def _create_random_block(self, idx):
        class MockLoader:
            def __init__(self, n_embd):
                self.n_embd = n_embd
            def get(self, name, required=True):
                if "norm" in name:
                    return type('obj', (object,), {'data': np.ones(self.n_embd).astype(np.float32)})
                return np.random.normal(0, 0.02, (self.n_embd, self.n_embd)).astype(np.float32)
        return GenomicTransformerBlock(MockLoader(self.n_embd), idx, self.n_head, self.n_head_kv, self.head_dim, self.rope_base, self.eps)

    def forward(self, tokens, clear_cache=True):
        if clear_cache:
            self.rust_llm.clear_cache()
            
        all_logits = []
        # Process each token sequentially to build KV cache correctly
        for tid in (tokens if isinstance(tokens, list) else [tokens]):
            # The Rust side now handles pos internally based on cache length
            logits = self.rust_llm.forward(tid, False) # Do NOT clear inside the loop
            all_logits.append(logits)
        return np.stack(all_logits)

    def generate(self, prompt, max_new_tokens=20, temperature=0.7, top_p=0.9, repetition_penalty=1.0):
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        generated_tokens = tokens.copy()
        
        # Inferencia inicial (prompt)
        next_token_logits = self.forward(tokens, clear_cache=True)[-1]
        
        for _ in range(max_new_tokens):
            # Debug: Check for numeric stability
            if np.isnan(next_token_logits).any() or np.isinf(next_token_logits).any():
                print("\n[!] WARNING: Logits explosion detected (NaN/Inf).")
                break

            # Repetition penalty
            penalized_logits = dna_semantic_compression.apply_repetition_penalty(
                next_token_logits.tolist(), 
                repetition_penalty, 
                generated_tokens[-20:]
            )
            
            # Muestreo Top-P
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
