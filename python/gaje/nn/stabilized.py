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
            if hasattr(bias_f32_or_tensor, 'tensor_type'):
                b_data = bias_f32_or_tensor.data
                if bias_f32_or_tensor.tensor_type == gguf.GGMLQuantizationType.F16:
                    self.bias = np.frombuffer(b_data, dtype=np.float16).astype(np.float32).tolist()
                else:
                    self.bias = np.frombuffer(b_data, dtype=np.float32).tolist()
            elif hasattr(bias_f32_or_tensor, 'tolist'):
                self.bias = bias_f32_or_tensor.tolist()
            else:
                self.bias = list(bias_f32_or_tensor)

        self.linear = dna_semantic_compression.GenomicLinear(
            self.dna_database, self.anchors_f16_bytes, self.dna_centroids, self.out_features, self.in_features, self.block_size,
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
        # Para evitar picos de memoria masivos (copias de matrices f32), procesamos por bloques
        # si la matriz es muy grande (> 100MB)
        matrix_size_mb = (weights_f32.nbytes) / (1024 * 1024)
        
        if matrix_size_mb > 100:
            print(f"    [~] Optimizando carga de capa pesada ({matrix_size_mb:.1f} MB)...")
            chunk_rows = 10000
            dna_db_list = []
            centroids_list = []
            anchors_list = []
            
            for i in range(0, self.out_features, chunk_rows):
                end = min(i + chunk_rows, self.out_features)
                w_chunk = weights_f32[i:end].copy()
                
                ret = dna_semantic_compression.genomize_f32_native(
                    w_chunk.tobytes(), 
                    block_size, 
                    anchor_threshold
                )
                d_db, d_c, a_bin = ret
                dna_db_list.append(d_db)
                centroids_list.extend(d_c)
                anchors_list.append(a_bin)
                del w_chunk
            
            self.dna_database = b"".join(dna_db_list)
            self.dna_centroids = centroids_list
            self.anchors_f16_bytes = b"".join(anchors_list)
        else:
            # Procedimiento estándar para capas pequeñas
            dna_db, dna_centroids, a_bin = dna_semantic_compression.genomize_f32_native(
                weights_f32.tobytes(), 
                block_size, 
                anchor_threshold
            )
            self.dna_database = dna_db
            self.dna_centroids = dna_centroids
            self.anchors_f16_bytes = a_bin
        
        self.epigenetic_database = b""
        self.epigenetic_centroids = []
        self.triplet_database = b""
        self.triplet_centroids = []
        
        if self.balancer:
            # El balancer también debería ser chunked si es necesario, 
            # pero por ahora lo dejamos así ya que suele usarse en capas internas
            stride = block_size // 4
            entropies = np.array(dna_semantic_compression.calculate_shannon_entropy(
                weights_f32.tobytes(), self.out_features, self.in_features
            ))
            full_mask = self.balancer.generate_precision_mask(entropies)
            self.precision_mask = [int(np.max(full_mask[b*block_size+k*4 : b*block_size+(k+1)*4])) 
                                   for b in range(self.in_features // block_size) 
                                   for k in range(stride)] * self.out_features
            
            self.epigenetic_database = b"\x00" * (len(self.dna_database))
            self.epigenetic_centroids = [0.0] * (len(self.dna_centroids))
            self.triplet_database = b"\x00" * (len(self.dna_database))
            self.triplet_centroids = [0.0] * (len(self.dna_centroids))
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
        
        # De-cuantizar anclas on-the-fly si es necesario para inspección
        anchors = np.frombuffer(self.anchors_f16_bytes, dtype=np.float16).astype(np.float32)
        return res + anchors[idx * self.in_features : (idx + 1) * self.in_features]

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
        act_fn = config.ffn_act if config else "swiglu"
        use_gen_norm = config.use_genomic_norm if config else False
        self.rust_block = dna_semantic_compression.RustGenomicBlock(
            idx, self.attn_layer.attn, self.attn_layer.q_gen.linear, self.attn_layer.k_gen.linear, 
            self.attn_layer.v_gen.linear, self.attn_layer.w_o.linear, 
            self.gate_gen.linear, self.up_gen.linear, self.w_down.linear, self.ffn_norm, self.eps,
            act_fn, use_gen_norm
        )

    def forward(self, x, pos):
        return np.array(self.rust_block.forward(x.tolist() if hasattr(x, "tolist") else x, pos), dtype=np.float32)

    def refine_ffn(self, x_norm, target, lr=0.01):
        """Native FFN refinement (SwiGLU, GeGLU, ReLU)."""
        self.rust_block.refine_ffn(x_norm.tolist() if hasattr(x_norm, "tolist") else x_norm, 
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
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_id)
        vocab_size = len(self.tokenizer)
        
        # Initialization logic
        if loader:
            embd_tensor = loader.get("token_embd.weight")
            self.embeddings = GenomicLayer("token_embd", embd_tensor, balancer=None, anchor_threshold=-1.0, config=self.config)
            output_norm = loader.get("output_norm.weight").data.astype(np.float32).tolist()
            
            head_tensor = loader.get("output.weight", required=False)
            if head_tensor is None or head_tensor.name == embd_tensor.name:
                print("    [*] Compartiendo pesos entre Embeddings y LM Head...")
                # Si comparten pesos, podemos reutilizar la lógica pero con distinto threshold?
                # En realidad, si comparten, solemos querer el mismo threshold o simplemente
                # re-genomizar pero sin cargar el tensor original dos veces.
                self.lm_head = GenomicLayer("lm_head", head_tensor or embd_tensor, balancer=None, anchor_threshold=0.1, config=self.config)
            else:
                self.lm_head = GenomicLayer("lm_head", head_tensor, balancer=None, anchor_threshold=0.1, config=self.config)
        else:
            emb_w = np.random.normal(0, 0.02, (vocab_size, self.n_embd)).astype(np.float32) 
            self.embeddings = GenomicLayer("token_embd", emb_w, balancer=None, anchor_threshold=-1.0, config=self.config)
            output_norm = np.ones(self.n_embd).astype(np.float32).tolist()
            lm_head_w = np.random.normal(0, 0.02, (vocab_size, self.n_embd)).astype(np.float32)
            self.lm_head = GenomicLayer("lm_head", lm_head_w, balancer=None, anchor_threshold=0.1, config=self.config)
        
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
            self.embeddings.linear, rust_blocks, output_norm, self.lm_head.linear, self.eps
        )
        
        end_total = time.time()
        print(f"[*] Sincronización Genómica Nativa finalizada en {end_total - start_total:.2f}s")

    def _create_random_block(self, idx):
        class MockLoader:
            def __init__(self, n_embd):
                self.n_embd = n_embd
            def get(self, name, required=True):
                if not required and "bias" in name:
                    return np.zeros(self.n_embd).astype(np.float32)
                if "norm" in name:
                    # Simulation of GGUF tensor object
                    return type('obj', (object,), {'data': np.ones(self.n_embd).astype(np.float32), 'tensor_type': 0}) 
                return np.random.normal(0, 0.02, (self.n_embd, self.n_embd)).astype(np.float32)
        return GenomicTransformerBlock(MockLoader(self.n_embd), idx, self.n_head, self.n_head_kv, self.head_dim, self.rope_base, self.eps, config=self.config)


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

    def save(self, output_path):
        """Saves the entire genomic organism to a single .gaje database."""
        import json
        import os
        import tempfile
        
        if not output_path.endswith('.gaje'):
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            output_path = os.path.join(output_path, "model.gaje")
            
        db_writer = dna_semantic_compression.GajeDatabaseWriter(output_path)
            
        # Save metadata
        metadata = {
            "config": {
                "name": self.config.name,
                "tokenizer_id": self.config.tokenizer_id,
                "rope_base": self.rope_base,
                "ffn_act": self.config.ffn_act,
                "use_genomic_norm": self.config.use_genomic_norm
            },
            "n_embd": self.n_embd,
            "n_head": self.n_head,
            "n_head_kv": self.n_head_kv,
            "n_blocks": self.n_blocks,
            "vocab_size": len(self.tokenizer) if hasattr(self, 'tokenizer') else 50257,
            "eps": self.eps
        }
        db_writer.write_metadata("config", json.dumps(metadata))
        
        # Save Tokenizer
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            if hasattr(self.tokenizer, 'is_fast') and self.tokenizer.is_fast and hasattr(self.tokenizer, 'backend_tokenizer'):
                tokenizer_str = self.tokenizer.backend_tokenizer.to_str()
                db_writer.write_metadata("tokenizer", tokenizer_str)
            else:
                with tempfile.TemporaryDirectory() as tmpdirname:
                    self.tokenizer.save_pretrained(tmpdirname)
                    tok_path = os.path.join(tmpdirname, "tokenizer.json")
                    if os.path.exists(tok_path):
                        with open(tok_path, "r", encoding="utf-8") as f:
                            db_writer.write_metadata("tokenizer", f.read())
            
        def save_layer(layer, name):
            db_writer.write_tensor(f"{name}.dna", np.frombuffer(layer.linear.database, dtype=np.uint8).tobytes())
            db_writer.write_tensor(f"{name}.centroids", np.array(layer.linear.centroids, dtype=np.float32).tobytes())
            # Usamos el atributo guardado durante la inicialización para evitar la reconstrucción de la lista
            db_writer.write_tensor(f"{name}.anchors", layer.anchors_f16_bytes)
            if hasattr(layer.linear, 'bias') and len(layer.linear.bias) > 0:
                db_writer.write_tensor(f"{name}.bias", np.array(layer.linear.bias, dtype=np.float32).tobytes())
                
            if hasattr(layer.linear, 'precision_mask') and len(layer.linear.precision_mask) > 0:
                db_writer.write_tensor(f"{name}.precision_mask", np.frombuffer(layer.linear.precision_mask, dtype=np.uint8).tobytes())
                db_writer.write_tensor(f"{name}.epi_dna", np.frombuffer(layer.linear.epigenetic_database, dtype=np.uint8).tobytes())
                db_writer.write_tensor(f"{name}.epi_centroids", np.array(layer.linear.epigenetic_centroids, dtype=np.float32).tobytes())
                db_writer.write_tensor(f"{name}.tri_dna", np.frombuffer(layer.linear.triplet_database, dtype=np.uint8).tobytes())
                db_writer.write_tensor(f"{name}.tri_centroids", np.array(layer.linear.triplet_centroids, dtype=np.float32).tobytes())

        # Save Embeddings
        save_layer(self.embeddings, "token_embd")
        # Save LM Head
        save_layer(self.lm_head, "lm_head")
        
        # Save Global Output Norm
        if hasattr(self.rust_llm, 'output_norm'):
            db_writer.write_tensor("output_norm", np.array(self.rust_llm.output_norm, dtype=np.float32).tobytes())
        
        # Save blocks
        for i, block in enumerate(self.blocks):
            p = f"blk.{i}."
            save_layer(block.attn_layer.q_gen, p + "attn_q")
            save_layer(block.attn_layer.k_gen, p + "attn_k")
            save_layer(block.attn_layer.v_gen, p + "attn_v")
            save_layer(block.attn_layer.w_o, p + "attn_output")
            save_layer(block.gate_gen, p + "ffn_gate")
            save_layer(block.up_gen, p + "ffn_up")
            save_layer(block.w_down, p + "ffn_down")
            
            # Save Block Norms
            if hasattr(block.rust_block, 'ffn_norm'):
                db_writer.write_tensor(p + "ffn_norm", np.array(block.rust_block.ffn_norm, dtype=np.float32).tobytes())
            if hasattr(block.rust_block, 'attn') and hasattr(block.rust_block.attn, 'rmsnorm_weight'):
                db_writer.write_tensor(p + "attn_norm", np.array(block.rust_block.attn.rmsnorm_weight, dtype=np.float32).tobytes())
            
        print(f"📦 Organismo genómico guardado en: {output_path}")

    @classmethod
    def load_genomic(cls, input_path):
        """Loads a previously saved genomic organism from a .gaje database."""
        import json
        import os
        import time
        from gaje.nn.configs import ArchitectureConfig
        
        if not input_path.endswith('.gaje'):
            input_path = os.path.join(input_path, "model.gaje")
            
        db_reader = dna_semantic_compression.GajeDatabaseReader(input_path)
        meta_str = db_reader.read_metadata("config")
        meta = json.loads(meta_str)
            
        config = ArchitectureConfig(**meta["config"])
        
        # Instantiate model directly without `__init__` calling random generation
        model = cls.__new__(cls)
        model.config = config
        model.n_embd = meta["n_embd"]
        model.n_head = meta["n_head"]
        model.n_head_kv = meta["n_head_kv"]
        model.head_dim = model.n_embd // model.n_head
        model.n_blocks = meta["n_blocks"]
        model.eps = meta["eps"]
        model.rope_base = meta["config"]["rope_base"]
        
        print(f"🧬 Despertando Organismo GAJE desde base de datos: {input_path}")
        start_total = time.time()

        def load_linear(name, out_features, in_features):
            dna = db_reader.read_tensor(f"{name}.dna")
            centroids = np.frombuffer(db_reader.read_tensor(f"{name}.centroids"), dtype=np.float32).tolist()
            # Pasamos bytes de anclas directamente
            anchors_u8 = db_reader.read_tensor(f"{name}.anchors")
            
            bias = []
            if db_reader.has_tensor(f"{name}.bias"):
                bias = np.frombuffer(db_reader.read_tensor(f"{name}.bias"), dtype=np.float32).tolist()
                
            precision_mask = []
            epi_dna = b""
            epi_centroids = []
            tri_dna = b""
            tri_centroids = []
            
            if db_reader.has_tensor(f"{name}.precision_mask"):
                precision_mask = list(db_reader.read_tensor(f"{name}.precision_mask"))
                epi_dna = db_reader.read_tensor(f"{name}.epi_dna")
                epi_centroids = np.frombuffer(db_reader.read_tensor(f"{name}.epi_centroids"), dtype=np.float32).tolist()
                tri_dna = db_reader.read_tensor(f"{name}.tri_dna")
                tri_centroids = np.frombuffer(db_reader.read_tensor(f"{name}.tri_centroids"), dtype=np.float32).tolist()
            
            linear = dna_semantic_compression.GenomicLinear(
                dna, anchors_u8, centroids, out_features, in_features, 32, 
                bias=bias,
                precision_mask=precision_mask,
                epigenetic_database=epi_dna,
                epigenetic_centroids=epi_centroids,
                triplet_database=tri_dna,
                triplet_centroids=tri_centroids
            )
            
            # Create a mock wrapper for Python interface
            class MockLayer:
                def __init__(self, lin, anchors_bin):
                    self.linear = lin
                    self.block_size = 32
                    self.anchors_f16_bytes = anchors_bin
                def forward(self, x):
                    return np.array(self.linear.forward(x.tolist() if hasattr(x, "tolist") else x), dtype=np.float32)
            
            return MockLayer(linear, anchors_u8)
            
        model.embeddings = load_linear("token_embd", meta.get("vocab_size", 50257), model.n_embd)
        model.lm_head = load_linear("lm_head", meta.get("vocab_size", 50257), model.n_embd)
        
        output_norm = np.ones(model.n_embd).astype(np.float32).tolist() 
        if db_reader.has_tensor("output_norm"):
            output_norm = np.frombuffer(db_reader.read_tensor("output_norm"), dtype=np.float32).tolist()
        
        rust_blocks = []
        model.blocks = []
        for i in range(model.n_blocks):
            p = f"blk.{i}."
            q_gen = load_linear(p + "attn_q", model.n_head * model.head_dim, model.n_embd)
            k_gen = load_linear(p + "attn_k", model.n_head_kv * model.head_dim, model.n_embd)
            v_gen = load_linear(p + "attn_v", model.n_head_kv * model.head_dim, model.n_embd)
            w_o = load_linear(p + "attn_output", model.n_embd, model.n_head * model.head_dim)
            
            # Use metadata or heuristics to determine FFN size
            # For Qwen/SmolLM, FFN is usually hidden_dim * (8/3) or similar
            # We check the centroids size to be sure
            def get_out_features(name, in_features):
                if not db_reader.has_tensor(f"{name}.centroids"): return model.n_embd * 4
                c_bytes = db_reader.read_tensor(f"{name}.centroids")
                c_count = len(c_bytes) // 4
                # Genomic linear: centroids count = out_features * (in_features // block_size) * 4
                return c_count // (in_features // 32 * 4)

            ffn_hidden = get_out_features(p + "ffn_gate", model.n_embd)
            
            gate_gen = load_linear(p + "ffn_gate", ffn_hidden, model.n_embd)
            up_gen = load_linear(p + "ffn_up", ffn_hidden, model.n_embd)
            w_down = load_linear(p + "ffn_down", model.n_embd, ffn_hidden)
            
            attn_norm_data = np.ones(model.n_embd).astype(np.float32).tolist()
            if db_reader.has_tensor(p + "attn_norm"):
                attn_norm_data = np.frombuffer(db_reader.read_tensor(p + "attn_norm"), dtype=np.float32).tolist()
                
            ffn_norm_data = np.ones(model.n_embd).astype(np.float32).tolist()
            if db_reader.has_tensor(p + "ffn_norm"):
                ffn_norm_data = np.frombuffer(db_reader.read_tensor(p + "ffn_norm"), dtype=np.float32).tolist()
            
            attn = dna_semantic_compression.GenomicAttention(model.n_head, model.n_head_kv, model.head_dim, attn_norm_data, model.eps, model.rope_base)
            
            act_fn = model.config.ffn_act if model.config else "swiglu"
            use_gen_norm = model.config.use_genomic_norm if model.config else False
            
            rust_block = dna_semantic_compression.RustGenomicBlock(
                i, attn, q_gen.linear, k_gen.linear, v_gen.linear, w_o.linear,
                gate_gen.linear, up_gen.linear, w_down.linear, ffn_norm_data, model.eps,
                act_fn, use_gen_norm
            )
            
            class MockBlock:
                def __init__(self, rb, q, k, v, o, gate, up, down):
                    self.rust_block = rb
                    self.attn_layer = type('obj', (object,), {'q_gen': q, 'k_gen': k, 'v_gen': v, 'w_o': o})
                    self.gate_gen = gate
                    self.up_gen = up
                    self.w_down = down
            
            model.blocks.append(MockBlock(rust_block, q_gen, k_gen, v_gen, w_o, gate_gen, up_gen, w_down))
            rust_blocks.append(rust_block)
            
        model.rust_llm = dna_semantic_compression.RustGenomicLLM(
            model.embeddings.linear, rust_blocks, output_norm, model.lm_head.linear, model.eps
        )
        
        # Carga de Tokenizador Soberana (desde la BD si es posible)
        from transformers import AutoTokenizer, PreTrainedTokenizerFast
        try:
            if db_reader.has_metadata("tokenizer"):
                import tempfile
                import json
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                    tmp.write(db_reader.read_metadata("tokenizer"))
                    tmp_path = tmp.name
                
                # Cargar como un tokenizador rápido directamente desde el archivo JSON
                model.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tmp_path)
                os.unlink(tmp_path)
                print("[*] Tokenizador cargado desde la base de datos genómica.")
            else:
                model.tokenizer = AutoTokenizer.from_pretrained(model.config.tokenizer_id)
        except Exception as e:
            print(f"[!] Aviso: Fallo al cargar tokenizador soberano, reintentando con ID: {e}")
            model.tokenizer = AutoTokenizer.from_pretrained(model.config.tokenizer_id)
        
        end_total = time.time()
        print(f"[*] Reconstrucción desde BD finalizada en {end_total - start_total:.2f}s")
        return model
