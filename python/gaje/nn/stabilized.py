import os
import numpy as np
import gaje.core._impl as dna_semantic_compression
import gguf
import time
from transformers import AutoTokenizer

from gaje.utils.quantization import dequantize_q8_0


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
    def __init__(
        self,
        name,
        weights_f32_or_tensor,
        bias_f32_or_tensor=None,
        block_size=32,
        anchor_threshold=None,
        rmsnorm_weight=None,
        eps=1e-6,
        balancer=None,
        n_head=None,
        head_dim=None,
        config=None,
        custom_base_c=None,
    ):
        self.name = name
        self.block_size = block_size
        self.balancer = balancer
        self.config = config
        self.custom_base_c = custom_base_c

        # Use config defaults if not provided
        if anchor_threshold is None:
            anchor_threshold = self.config.anchor_threshold if self.config else -1.0

        is_q_or_k = "attn_q" in name or "attn_k" in name
        # También consideramos attn_v y attn_output para 4-bit en Mixed-Bit strategy
        # Para el test de rescate, probamos F32 en TODO para aislar errores del motor.
        bit_depth = 32

        if hasattr(weights_f32_or_tensor, "tensor_type"):
            tensor = weights_f32_or_tensor

            # GGUF tensors are usually [in_features, out_features] or similar
            if len(tensor.shape) == 2:
                self.in_features, self.out_features = tensor.shape
            else:
                self.out_features, self.in_features = tensor.shape[0], 1

            if tensor.tensor_type in [
                gguf.GGMLQuantizationType.F16,
                gguf.GGMLQuantizationType.F32,
            ]:
                if tensor.tensor_type == gguf.GGMLQuantizationType.F16:
                    raw_data = np.frombuffer(tensor.data, dtype=np.float16).astype(
                        np.float32
                    )
                else:
                    raw_data = np.frombuffer(tensor.data, dtype=np.float32)

                # GGUF: first dim is fastest. Correct reshape is [out, in]
                w_matrix = raw_data.reshape(self.out_features, self.in_features)

                unpermute = self.config.unpermute_weights if self.config else True
                needs_unpermute = (
                    unpermute
                    and is_q_or_k
                    and n_head is not None
                    and head_dim is not None
                    and (self.config.rope_style == "split" if self.config else True)
                )
                print(
                    f"    [~] Capa {name}: Unpermute={unpermute}, Needs-Unpermute={needs_unpermute}, Bit-Depth={bit_depth}"
                )

                if (
                    bit_depth
                    != 32  # Si es F32, forzamos el path lento para tener f32_data
                    and not needs_unpermute
                    and tensor.tensor_type == gguf.GGMLQuantizationType.F16
                ):
                    # DGI Fast Path: Direct streaming conversion in Rust
                    (
                        dna,
                        centroids,
                        anchors,
                    ) = dna_semantic_compression.genomize_f16_native(
                        tensor.data.tobytes(),
                        block_size,
                        anchor_threshold,
                        bit_depth,  # Pass bit_depth
                        custom_base_c,
                    )
                    self.dna_database = bytes(dna) if isinstance(dna, list) else dna
                    self.dna_centroids = centroids
                    self.anchors_f16_bytes = (
                        bytes(anchors) if isinstance(anchors, list) else anchors
                    )
                    self.bit_depth = bit_depth  # Save bit_depth

                    self.epigenetic_database = b""
                    self.epigenetic_centroids = []
                    self.triplet_database = b""
                    self.triplet_centroids = []
                    self.precision_mask = []
                else:
                    if tensor.tensor_type == gguf.GGMLQuantizationType.F16:
                        raw_data = np.frombuffer(tensor.data, dtype=np.float16).astype(
                            np.float32
                        )
                    else:
                        raw_data = np.frombuffer(tensor.data, dtype=np.float32)

                    # GGUF: first dim is fastest. Correct reshape is [out, in]
                    w_matrix = raw_data.reshape(self.out_features, self.in_features)

                    if needs_unpermute:
                        from gaje.utils.quantization import unpermute_to_split

                        w_matrix = unpermute_to_split(w_matrix, n_head, head_dim)

                    self._init_from_f32(
                        w_matrix, block_size, anchor_threshold, custom_base_c, bit_depth
                    )
            else:
                # Solo para Q8_0 aplicamos la lógica de des-permutación necesaria
                rope_style = self.config.rope_style if self.config else "split"
                weights_f32 = dequantize_q8_0(
                    tensor, n_head, head_dim, is_q_or_k, rope_style=rope_style
                )

                unpermute = self.config.unpermute_weights if self.config else True
                needs_unpermute = (
                    unpermute
                    and is_q_or_k
                    and n_head is not None
                    and head_dim is not None
                    and (self.config.rope_style == "split" if self.config else True)
                )
                if needs_unpermute:
                    from gaje.utils.quantization import unpermute_to_split

                    weights_f32 = unpermute_to_split(weights_f32, n_head, head_dim)

                self.out_features, self.in_features = weights_f32.shape
                self._init_from_f32(
                    weights_f32, block_size, anchor_threshold, custom_base_c, bit_depth
                )
        else:
            weights_f32 = weights_f32_or_tensor
            self.out_features, self.in_features = weights_f32.shape

            # También para pesos brutos (born-genomic o manual), respetamos el flag
            unpermute = (
                self.config.unpermute_weights if self.config else False
            )  # Default false for raw
            rope_style = self.config.rope_style if self.config else "split"
            if (
                unpermute
                and is_q_or_k
                and n_head is not None
                and head_dim is not None
                and rope_style != "split"
            ):
                from gaje.utils.quantization import unpermute_to_interleaved

                weights_f32 = unpermute_to_interleaved(weights_f32, n_head, head_dim)

            self._init_from_f32(
                weights_f32, block_size, anchor_threshold, custom_base_c, bit_depth
            )

        # Manejo de Bias (Crucial para Qwen2)
        self.bias = []
        if bias_f32_or_tensor is not None:
            if hasattr(bias_f32_or_tensor, "tensor_type"):
                b_data = bias_f32_or_tensor.data
                if bias_f32_or_tensor.tensor_type == gguf.GGMLQuantizationType.F16:
                    self.bias = (
                        np.frombuffer(b_data, dtype=np.float16)
                        .astype(np.float32)
                        .tolist()
                    )
                else:
                    self.bias = np.frombuffer(b_data, dtype=np.float32).tolist()
            elif hasattr(bias_f32_or_tensor, "tolist"):
                self.bias = bias_f32_or_tensor.tolist()
            else:
                self.bias = list(bias_f32_or_tensor)

        self.linear = dna_semantic_compression.GenomicLinear(
            self.dna_database,
            self.anchors_f16_bytes,
            self.dna_centroids,
            self.out_features,
            self.in_features,
            self.block_size,
            rmsnorm_weight=rmsnorm_weight.tolist()
            if rmsnorm_weight is not None
            else [],
            eps=eps,
            precision_mask=self.precision_mask,
            epigenetic_database=self.epigenetic_database,
            epigenetic_centroids=self.epigenetic_centroids,
            triplet_database=self.triplet_database,
            triplet_centroids=self.triplet_centroids,
            bias=self.bias,
            bit_depth=self.bit_depth,
        )

    def _init_from_f32(
        self, weights_f32, block_size, anchor_threshold, custom_base_c=None, bit_depth=2
    ):
        # Para evitar picos de memoria masivos (copias de matrices f32), procesamos por bloques
        # si la matriz es muy grande (> 100MB)
        matrix_size_mb = (weights_f32.nbytes) / (1024 * 1024)

        if matrix_size_mb > 100:
            print(
                f"    [~] Optimizando carga de capa pesada ({matrix_size_mb:.1f} MB)..."
            )
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
                    anchor_threshold,
                    bit_depth,
                    custom_base_c,
                )
                d_db, d_c, a_bin = ret
                dna_db_list.append(d_db)
                centroids_list.extend(d_c)
                anchors_list.append(a_bin)
                del w_chunk

            if not dna_db_list:
                self.dna_database = b""
                self.dna_centroids = []
                self.anchors_f16_bytes = b""
            else:
                self.dna_database = b"".join(dna_db_list)
                self.dna_centroids = centroids_list
                self.anchors_f16_bytes = b"".join(anchors_list)
        else:
            # Procedimiento estándar para capas pequeñas
            dna_db, dna_centroids, a_bin = dna_semantic_compression.genomize_f32_native(
                weights_f32.tobytes(),
                block_size,
                anchor_threshold,
                bit_depth,
                custom_base_c,
            )
            self.dna_database = dna_db
            self.dna_centroids = dna_centroids
            self.anchors_f16_bytes = a_bin

        self.bit_depth = bit_depth

        self.epigenetic_database = b""
        self.epigenetic_centroids = []
        self.triplet_database = b""
        self.triplet_centroids = []

        if self.balancer:
            # El balancer también debería ser chunked si es necesario,
            # pero por ahora lo dejamos así ya que suele usarse en capas internas
            stride = block_size // 4
            entropies = np.array(
                dna_semantic_compression.calculate_shannon_entropy(
                    weights_f32.tobytes(), self.out_features, self.in_features
                )
            )
            full_mask = self.balancer.generate_precision_mask(entropies)
            self.precision_mask = [
                int(
                    np.max(
                        full_mask[b * block_size + k * 4 : b * block_size + (k + 1) * 4]
                    )
                )
                for b in range(self.in_features // block_size)
                for k in range(stride)
            ] * self.out_features

            self.epigenetic_database = b"\x00" * (len(self.dna_database))
            self.epigenetic_centroids = [0.0] * (len(self.dna_centroids))
            self.triplet_database = b"\x00" * (len(self.dna_database))
            self.triplet_centroids = [0.0] * (len(self.dna_centroids))
        else:
            self.precision_mask = []

    def forward(self, x):
        return np.array(
            self.linear.forward(x.tolist() if hasattr(x, "tolist") else x, False),
            dtype=np.float32,
        )

    def refine_centroids(self, x, target, lr=0.01):
        """Refines the centroids of this layer based on input and target output."""
        self.linear.refine_centroids(
            x.tolist() if hasattr(x, "tolist") else x,
            target.tolist() if hasattr(target, "tolist") else target,
            lr,
        )

    def refine_with_grads(self, x, grads, lr=0.01):
        """Refines the centroids of this layer based on input and gradients."""
        self.linear.refine_with_grads(
            x.tolist() if hasattr(x, "tolist") else x,
            grads.tolist() if hasattr(grads, "tolist") else grads,
            lr,
        )

    def get_row(self, idx):
        # Infer stride and mapping from bit_depth
        bit_depth = getattr(self, "bit_depth", 2)
        if bit_depth == 32:
            row_start = idx * self.in_features * 4
            row_bytes = self.dna_database[row_start : row_start + self.in_features * 4]
            return np.frombuffer(row_bytes, dtype=np.float32).copy()

        if bit_depth == 4:
            n_blocks = self.in_features // self.block_size
            stride = self.block_size // 2
            dna_row = self.dna_database[
                idx * n_blocks * stride : (idx + 1) * n_blocks * stride
            ]
            res = np.zeros(self.in_features, dtype=np.float32)
            for b in range(n_blocks):
                block_dna = dna_row[b * stride : (b + 1) * stride]
                block_centroids = self.dna_centroids[
                    (idx * n_blocks + b) * 16 : (idx * n_blocks + b) * 16 + 16
                ]
                for k in range(stride):
                    byte = block_dna[k]
                    # High nibble
                    res[b * self.block_size + k * 2] = block_centroids[
                        (byte >> 4) & 0x0F
                    ]
                    # Low nibble
                    res[b * self.block_size + k * 2 + 1] = block_centroids[byte & 0x0F]
        else:
            n_blocks, stride = self.in_features // self.block_size, self.block_size // 4
            dna_row = self.dna_database[
                idx * n_blocks * stride : (idx + 1) * n_blocks * stride
            ]
            res = np.zeros(self.in_features, dtype=np.float32)
            for b in range(n_blocks):
                res[b * self.block_size : (b + 1) * self.block_size] = np.array(
                    dna_semantic_compression.dequantize_embedding(
                        dna_row[b * stride : (b + 1) * stride],
                        self.block_size,
                        self.dna_centroids[
                            (idx * n_blocks + b) * 4 : (idx * n_blocks + b) * 4 + 4
                        ],
                    )
                )

        # De-cuantizar anclas on-the-fly si es necesario para inspección
        if len(self.anchors_f16_bytes) < 8:  # Mínimo "GAJE" + count
            return res

        anchors = np.frombuffer(self.anchors_f16_bytes, dtype=np.float16).astype(
            np.float32
        )
        # Nota: Esto asume anclas densas o un formato compatible para get_row.
        # Para GAJE real (sparse), get_row debería usar la lógica de reconstrucción nativa.
        if len(anchors) == 0:
            return res

        try:
            return res + anchors[idx * self.in_features : (idx + 1) * self.in_features]
        except:
            return res


class GenomicAttentionLayer:
    def __init__(
        self,
        loader,
        p,
        n_head,
        n_head_kv,
        head_dim,
        rope_base,
        rmsnorm_weight=None,
        eps=1e-6,
        config=None,
    ):
        self.config = config
        self.q_gen = GenomicLayer(
            p + "attn_q",
            loader.get(p + "attn_q.weight"),
            bias_f32_or_tensor=loader.get(p + "attn_q.bias", required=False),
            n_head=n_head,
            head_dim=head_dim,
            balancer=None,
            anchor_threshold=self.config.anchor_threshold,
            config=config,
        )
        self.k_gen = GenomicLayer(
            p + "attn_k",
            loader.get(p + "attn_k.weight"),
            bias_f32_or_tensor=loader.get(p + "attn_k.bias", required=False),
            n_head=n_head_kv,
            head_dim=head_dim,
            balancer=None,
            anchor_threshold=self.config.anchor_threshold,
            config=config,
        )
        self.v_gen = GenomicLayer(
            p + "attn_v",
            loader.get(p + "attn_v.weight"),
            bias_f32_or_tensor=loader.get(p + "attn_v.bias", required=False),
            balancer=None,
            anchor_threshold=self.config.anchor_threshold,
            config=config,
        )
        self.w_o = GenomicLayer(
            p + "attn_output",
            loader.get(p + "attn_output.weight"),
            bias_f32_or_tensor=loader.get(p + "attn_output.bias", required=False),
            balancer=None,
            anchor_threshold=self.config.anchor_threshold,
            config=config,
        )
        self.attn = dna_semantic_compression.GenomicAttention(
            n_head,
            n_head_kv,
            head_dim,
            rmsnorm_weight.tolist() if rmsnorm_weight is not None else [],
            eps,
            rope_base,
            config.rope_style if config else "split",
        )

    def forward(self, x, pos):
        x_norm = self.attn.apply_rmsnorm(x.tolist() if hasattr(x, "tolist") else x)
        q, k, v = (
            self.q_gen.forward(np.array(x_norm)),
            self.k_gen.forward(np.array(x_norm)),
            self.v_gen.forward(np.array(x_norm)),
        )
        return self.w_o.forward(
            np.array(
                self.attn.forward_attention(q.tolist(), k.tolist(), v.tolist(), pos)
            )
        )


class GenomicTransformerBlock:
    def __init__(
        self,
        loader,
        idx,
        n_head,
        n_head_kv,
        head_dim,
        rope_base,
        eps,
        anchor_threshold=None,
        ffn_anchor_threshold=None,
        config=None,
        custom_centroids=None,
    ):
        self.config = config

        # Use config defaults if not provided
        if anchor_threshold is None:
            anchor_threshold = self.config.anchor_threshold if self.config else -1.0
        if ffn_anchor_threshold is None:
            ffn_anchor_threshold = (
                self.config.ffn_anchor_threshold if self.config else -1.0
            )
        p = f"blk.{idx}."
        attn_norm_data = loader.get(p + "attn_norm.weight").data.astype(np.float32)

        # Resolve custom centroids for this block
        layer_c = custom_centroids or {}

        self.attn_layer = GenomicAttentionLayer(
            loader,
            p,
            n_head,
            n_head_kv,
            head_dim,
            rope_base,
            rmsnorm_weight=attn_norm_data,
            eps=eps,
            config=config,
        )
        # Update attn_q with custom centroids if available
        if p + "attn_q.weight" in layer_c:
            self.attn_layer.q_gen = GenomicLayer(
                p + "attn_q",
                loader.get(p + "attn_q.weight"),
                anchor_threshold=anchor_threshold,
                config=config,
                custom_base_c=layer_c[p + "attn_q.weight"],
            )

        g, u = loader.get(p + "ffn_gate.weight"), loader.get(p + "ffn_up.weight")
        self.gate_gen, self.up_gen = (
            GenomicLayer(
                p + "gate",
                g,
                balancer=None,
                anchor_threshold=ffn_anchor_threshold,
                config=config,
                custom_base_c=layer_c.get(p + "ffn_gate.weight"),
            ),
            GenomicLayer(
                p + "up",
                u,
                balancer=None,
                anchor_threshold=ffn_anchor_threshold,
                config=config,
                custom_base_c=layer_c.get(p + "ffn_up.weight"),
            ),
        )
        self.w_down = GenomicLayer(
            p + "ffn_down",
            loader.get(p + "ffn_down.weight"),
            balancer=None,
            anchor_threshold=ffn_anchor_threshold,
            config=config,
            custom_base_c=layer_c.get(p + "ffn_down.weight"),
        )
        self.ffn_norm = (
            loader.get(p + "ffn_norm.weight").data.astype(np.float32).tolist()
        )
        self.eps = eps

        # Reference to the Rust block
        act_fn = config.ffn_act if config else "swiglu"
        use_gen_norm = config.use_genomic_norm if config else False
        rna_threshold = config.rna_threshold if config else 0.5
        self.rust_block = dna_semantic_compression.RustGenomicBlock(
            idx,
            self.attn_layer.attn,
            self.attn_layer.q_gen.linear,
            self.attn_layer.k_gen.linear,
            self.attn_layer.v_gen.linear,
            self.attn_layer.w_o.linear,
            self.gate_gen.linear,
            self.up_gen.linear,
            self.w_down.linear,
            self.ffn_norm,
            self.eps,
            act_fn,
            use_gen_norm,
            1.0,  # h_scale default
            rna_threshold,
        )

    def forward(self, x, pos):
        return np.array(
            self.rust_block.forward(x.tolist() if hasattr(x, "tolist") else x, pos),
            dtype=np.float32,
        )

    def refine_ffn(self, x_norm, target, lr=0.01):
        """Native FFN refinement (SwiGLU, GeGLU, ReLU)."""
        self.rust_block.refine_ffn(
            x_norm.tolist() if hasattr(x_norm, "tolist") else x_norm,
            target.tolist() if hasattr(target, "tolist") else target,
            lr,
        )

    def refine_attention(self, x_norm, target, pos, lr=0.01):
        """Native attention projection refinement."""
        self.rust_block.refine_attention(
            x_norm.tolist() if hasattr(x_norm, "tolist") else x_norm,
            target.tolist() if hasattr(target, "tolist") else target,
            pos,
            lr,
        )


from gaje.nn.configs import get_config, detect_arch  # noqa: E402


class GenomicLLM:
    def __init__(
        self,
        model_path=None,
        num_blocks=None,
        config=None,
        n_embd=None,
        n_head=None,
        custom_centroids=None,
    ):
        start_total = time.time()

        if model_path:
            self.reader = gguf.GGUFReader(model_path)
            loader = TensorLoader(self.reader)
            arch_name = detect_arch(self.reader)
            self.config = config or get_config(arch_name)
            print(
                f"[*] Arquitectura detectada: {arch_name} (Usando config: {self.config.name})"
            )

            self.n_embd = int(
                self.reader.fields[f"{arch_name}.embedding_length"].parts[-1][0]
            )
            self.n_head = int(
                self.reader.fields[f"{arch_name}.attention.head_count"].parts[-1][0]
            )
            self.n_head_kv = int(
                self.reader.fields[f"{arch_name}.attention.head_count_kv"].parts[-1][0]
            )
            self.head_dim = self.n_embd // self.n_head
            self.n_blocks = (
                int(self.reader.fields[f"{arch_name}.block_count"].parts[-1][0])
                if num_blocks is None
                else num_blocks
            )
            self.eps = float(
                self.reader.fields[
                    f"{arch_name}.attention.layer_norm_rms_epsilon"
                ].parts[-1][0]
            )

            rope_base_field = f"{arch_name}.rope.freq_base"
            llama_rope_base_field = "llama.rope.freq_base"
            if rope_base_field in self.reader.fields:
                self.rope_base = float(self.reader.fields[rope_base_field].parts[-1][0])
            elif llama_rope_base_field in self.reader.fields:
                self.rope_base = float(
                    self.reader.fields[llama_rope_base_field].parts[-1][0]
                )
            else:
                self.rope_base = self.config.rope_base

            if self.config.apply_smollm_rope_patch and self.rope_base > 1000000.0:
                print(
                    f"[!] Aviso: Forzando RoPE Base a 10000.0 para {os.path.basename(model_path)}"
                )
                self.rope_base = 10000.0
        else:
            # Born-genomic initialization
            if config is None:
                raise ValueError("Must provide config for born-genomic initialization")
            self.config = config
            self.n_embd = n_embd or 768
            self.n_head = n_head or 12
            self.n_head_kv = self.n_head  # Default to matching n_head for born models
            self.head_dim = self.n_embd // self.n_head
            self.n_blocks = num_blocks or 12
            self.eps = 1e-6
            self.rope_base = self.config.rope_base
            loader = None
            print(
                f"🧬 Iniciando Organismo GAJE Nativo (Born-Genomic): {self.config.name}"
            )

        # Inyectar centroides desde la configuración si no se proveen manualmente
        self.custom_centroids = custom_centroids or (
            self.config.default_centroids
            if hasattr(self.config, "default_centroids")
            else {}
        )

        if self.custom_centroids:
            print(
                f"    [*] Aplicando calibración de fábrica ({len(self.custom_centroids)} capas)..."
            )

        print(f"[*] RoPE Base: {self.rope_base}")

        # 1. Configuración del Tokenizer
        if self.config and self.config.tokenizer_id:
            if os.path.exists(self.config.tokenizer_id):
                if os.path.isdir(self.config.tokenizer_id):
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.config.tokenizer_id
                    )
                elif self.config.tokenizer_id.endswith(".json"):
                    from tokenizers import Tokenizer as lib_tokenizer

                    self.tokenizer = lib_tokenizer.from_file(self.config.tokenizer_id)
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.config.tokenizer_id
                    )
            else:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.config.tokenizer_id
                    )
                except Exception as e:
                    print(
                        f"[!] Warning: Could not load tokenizer '{self.config.tokenizer_id}': {e}"
                    )
                    self.tokenizer = None
        else:
            self.tokenizer = None

        if self.tokenizer:
            if hasattr(self.tokenizer, "get_vocab_size"):
                vocab_size = self.tokenizer.get_vocab_size()
            else:
                vocab_size = len(self.tokenizer)
        else:
            vocab_size = 0

        # Initialization logic
        if loader:
            embd_tensor = loader.get("token_embd.weight")
            self.embeddings = GenomicLayer(
                "token_embd",
                embd_tensor,
                balancer=None,
                anchor_threshold=self.config.anchor_threshold,
                config=self.config,
            )
            output_norm = (
                loader.get("output_norm.weight").data.astype(np.float32).tolist()
            )

            head_tensor = loader.get("output.weight", required=False)
            if head_tensor is None or head_tensor.name == embd_tensor.name:
                print("    [*] Compartiendo pesos entre Embeddings y LM Head...")
                # Si comparten pesos, podemos reutilizar la lógica pero con distinto threshold?
                # En realidad, si comparten, solemos querer el mismo threshold o simplemente
                # re-genomizar pero sin cargar el tensor original dos veces.
                self.lm_head = GenomicLayer(
                    "lm_head",
                    head_tensor or embd_tensor,
                    balancer=None,
                    anchor_threshold=self.config.anchor_threshold,
                    config=self.config,
                )
            else:
                self.lm_head = GenomicLayer(
                    "lm_head",
                    head_tensor,
                    balancer=None,
                    anchor_threshold=self.config.anchor_threshold,
                    config=self.config,
                )
        else:
            emb_w = np.random.normal(0, 0.02, (vocab_size, self.n_embd)).astype(
                np.float32
            )
            self.embeddings = GenomicLayer(
                "token_embd",
                emb_w,
                balancer=None,
                anchor_threshold=self.config.anchor_threshold,
                config=self.config,
            )
            output_norm = np.ones(self.n_embd).astype(np.float32).tolist()
            lm_head_w = np.random.normal(0, 0.02, (vocab_size, self.n_embd)).astype(
                np.float32
            )
            self.lm_head = GenomicLayer(
                "lm_head",
                lm_head_w,
                balancer=None,
                anchor_threshold=self.config.anchor_threshold,
                config=self.config,
            )

        rust_blocks = []
        self.blocks = []
        for i in range(self.n_blocks):
            if loader:
                block = GenomicTransformerBlock(
                    loader,
                    i,
                    self.n_head,
                    self.n_head_kv,
                    self.head_dim,
                    self.rope_base,
                    self.eps,
                    anchor_threshold=self.config.anchor_threshold,
                    ffn_anchor_threshold=self.config.ffn_anchor_threshold,
                    config=self.config,
                    custom_centroids=custom_centroids,
                )
            else:
                block = self._create_random_block(i)

            self.blocks.append(block)
            rust_blocks.append(block.rust_block)
            if (i + 1) % 10 == 0:
                print(f"    [~] Bloque {i + 1}/{self.n_blocks} sincronizado...")

        self.rust_llm = dna_semantic_compression.RustGenomicLLM(
            self.embeddings.linear,
            rust_blocks,
            output_norm,
            self.lm_head.linear,
            self.eps,
        )

        end_total = time.time()
        print(
            f"[*] Sincronización Genómica Nativa finalizada en {end_total - start_total:.2f}s"
        )

    def _create_random_block(self, idx):
        class MockLoader:
            def __init__(self, n_embd):
                self.n_embd = n_embd

            def get(self, name, required=True):
                if not required and "bias" in name:
                    return np.zeros(self.n_embd).astype(np.float32)
                if "norm" in name:
                    # Simulation of GGUF tensor object
                    return type(
                        "obj",
                        (object,),
                        {
                            "data": np.ones(self.n_embd).astype(np.float32),
                            "tensor_type": 0,
                        },
                    )
                return np.random.normal(0, 0.02, (self.n_embd, self.n_embd)).astype(
                    np.float32
                )

        return GenomicTransformerBlock(
            MockLoader(self.n_embd),
            idx,
            self.n_head,
            self.n_head_kv,
            self.head_dim,
            self.rope_base,
            self.eps,
            config=self.config,
        )

    def clear_cache(self):
        self.rust_llm.clear_cache_py()

    def set_k_wta_ratio(self, ratio: float):
        if hasattr(self, "rust_llm") and self.rust_llm:
            self.rust_llm.set_k_wta_ratio(ratio)

    def forward(self, tokens, clear_cache=True):
        if clear_cache:
            self.rust_llm.clear_cache_py()

        all_logits = []
        # Process each token sequentially to build KV cache correctly
        for tid in tokens if isinstance(tokens, list) else [tokens]:
            # The Rust side now handles pos internally based on cache length
            logits = self.rust_llm.forward(tid, False)  # Do NOT clear inside the loop
            all_logits.append(logits)
        return np.stack(all_logits)

    def generate(
        self,
        prompt,
        max_new_tokens=20,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.0,
        use_spiking=False,
        spiking_steps=24,
        spiking_threshold=0.5,
        spiking_decay=0.8,
        use_toroidal=True,
        toroidal_mass=1.0,
        toroidal_curvature=0.1,
    ):
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        if hasattr(tokens, "ids"):
            tokens = tokens.ids

        generated_tokens = list(tokens)

        # Inicializar Sampler Toroidal si se solicita
        toroidal_sampler = None
        if use_toroidal and not use_spiking:
            toroidal_sampler = dna_semantic_compression.ToroidalSampler(
                toroidal_mass, toroidal_curvature
            )

        # Inferencia inicial (prompt)
        # Para el prompt, procesamos token por token
        if use_spiking:
            # Procesamos todos menos el último tradicionalmente para llenar caché
            for tid in tokens[:-1]:
                self.rust_llm.forward(tid, False)
            # El último token dispara el primer logit con spiking
            next_token_logits = self.rust_llm.forward_spiking(
                tokens[-1], spiking_steps, spiking_threshold, spiking_decay
            )
        else:
            next_token_logits = self.forward(tokens, clear_cache=True)[-1]

        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            # Try to find by common names
            for name in ["<|im_end|>", "<|endoftext|>", "</s>"]:
                try:
                    id_ = self.tokenizer.token_to_id(name)
                    if id_ is not None:
                        eos_token_id = id_
                        break
                except:
                    continue

        for _ in range(max_new_tokens):
            # Debug: Check for numeric stability
            if np.isnan(next_token_logits).any() or np.isinf(next_token_logits).any():
                print("\n[!] WARNING: Logits explosion detected (NaN/Inf).")
                break

            # Repetition penalty
            penalized_logits = dna_semantic_compression.apply_repetition_penalty(
                next_token_logits.tolist()
                if hasattr(next_token_logits, "tolist")
                else list(next_token_logits),
                repetition_penalty,
                generated_tokens[-20:],
            )

            # Muestreo: Toroidal o Top-P tradicional
            if toroidal_sampler:
                next_id = toroidal_sampler.sample(penalized_logits, temperature, top_p)
            else:
                next_id = dna_semantic_compression.sample_top_p(
                    penalized_logits, top_p, temperature
                )

            if eos_token_id is not None and next_id == eos_token_id:
                break

            generated_tokens.append(next_id)
            yield self.tokenizer.decode([next_id])

            # Siguiente paso de inferencia (incremental)
            if use_spiking:
                next_token_logits = self.rust_llm.forward_spiking(
                    next_id, spiking_steps, spiking_threshold, spiking_decay
                )
            else:
                next_token_logits = self.rust_llm.forward(
                    next_id, False
                )  # Direct call to rust_llm

    def save(self, output_path):
        """Saves the entire genomic organism to a single .gaje database."""
        import json
        import os
        import tempfile
        from gaje.utils.version import get_project_version

        if not output_path.endswith(".gaje"):
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            output_path = os.path.join(output_path, "model.gaje")

        db_writer = dna_semantic_compression.GajeDatabaseWriter(output_path)

        # Save metadata
        metadata = {
            "config": {
                "name": self.config.name,
                "version": get_project_version(),
                "tokenizer_id": self.config.tokenizer_id,
                "rope_base": self.rope_base,
                "ffn_act": self.config.ffn_act,
                "use_genomic_norm": self.config.use_genomic_norm,
            },
            "n_embd": self.n_embd,
            "n_head": self.n_head,
            "n_head_kv": self.n_head_kv,
            "n_blocks": self.n_blocks,
            "vocab_size": len(self.tokenizer) if hasattr(self, "tokenizer") else 50257,
            "eps": self.eps,
        }
        db_writer.write_metadata("config", json.dumps(metadata))

        # Save Tokenizer
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            if (
                hasattr(self.tokenizer, "is_fast")
                and self.tokenizer.is_fast
                and hasattr(self.tokenizer, "backend_tokenizer")
            ):
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
            db_writer.write_tensor_compressed(
                f"{name}.dna",
                np.frombuffer(layer.linear.database, dtype=np.uint8).tobytes(),
            )
            db_writer.write_tensor_compressed(
                f"{name}.centroids",
                np.array(layer.linear.centroids, dtype=np.float32).tobytes(),
            )
            # Usamos el atributo guardado durante la inicialización para evitar la reconstrucción de la lista
            db_writer.write_tensor_compressed(
                f"{name}.anchors", layer.anchors_f16_bytes
            )
            if hasattr(layer.linear, "bias") and len(layer.linear.bias) > 0:
                db_writer.write_tensor_compressed(
                    f"{name}.bias",
                    np.array(layer.linear.bias, dtype=np.float32).tobytes(),
                )

            if (
                hasattr(layer.linear, "precision_mask")
                and len(layer.linear.precision_mask) > 0
            ):
                db_writer.write_tensor_compressed(
                    f"{name}.precision_mask",
                    np.frombuffer(
                        layer.linear.precision_mask, dtype=np.uint8
                    ).tobytes(),
                )
                db_writer.write_tensor_compressed(
                    f"{name}.epi_dna",
                    np.frombuffer(
                        layer.linear.epigenetic_database, dtype=np.uint8
                    ).tobytes(),
                )
                db_writer.write_tensor_compressed(
                    f"{name}.epi_centroids",
                    np.array(
                        layer.linear.epigenetic_centroids, dtype=np.float32
                    ).tobytes(),
                )
                db_writer.write_tensor_compressed(
                    f"{name}.tri_dna",
                    np.frombuffer(
                        layer.linear.triplet_database, dtype=np.uint8
                    ).tobytes(),
                )
                db_writer.write_tensor_compressed(
                    f"{name}.tri_centroids",
                    np.array(
                        layer.linear.triplet_centroids, dtype=np.float32
                    ).tobytes(),
                )

        # Save Embeddings
        save_layer(self.embeddings, "token_embd")
        # Save LM Head
        save_layer(self.lm_head, "lm_head")

        # Save Global Output Norm
        if hasattr(self.rust_llm, "output_norm"):
            db_writer.write_tensor_compressed(
                "output_norm",
                np.array(self.rust_llm.output_norm, dtype=np.float32).tobytes(),
            )

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
            if hasattr(block.rust_block, "ffn_norm"):
                db_writer.write_tensor_compressed(
                    p + "ffn_norm",
                    np.array(block.rust_block.ffn_norm, dtype=np.float32).tobytes(),
                )
            if hasattr(block.rust_block, "attn") and hasattr(
                block.rust_block.attn, "rmsnorm_weight"
            ):
                db_writer.write_tensor_compressed(
                    p + "attn_norm",
                    np.array(
                        block.rust_block.attn.rmsnorm_weight, dtype=np.float32
                    ).tobytes(),
                )

        print(f"📦 Organismo genómico guardado en: {output_path}")

    @classmethod
    def load_genomic(cls, input_path):
        """Loads a previously saved genomic organism from a .gaje database."""
        import json
        import os
        import time
        from gaje.nn.configs import ArchitectureConfig
        from gaje.nn import constants as C

        if not input_path.endswith(".gaje"):
            input_path = os.path.join(input_path, "model.gaje")

        db_reader = dna_semantic_compression.GajeDatabaseReader(input_path)

        # Try to read metadata safely
        try:
            meta_str = db_reader.read_metadata(C.META_KEY_CONFIG)
            meta = json.loads(meta_str)
        except Exception:
            # LEGACY FALLBACK: If 'config' is missing, the model is likely SMG1 or older.
            print(
                f"⚠️ Warning: Model at {input_path} lacks modern metadata. Applying legacy recovery..."
            )
            meta = {
                C.META_KEY_CONFIG: {
                    "name": "legacy_recovered",
                    "version": "unknown",
                    "rope_base": C.DEFAULT_ROPE_BASE,
                },
                C.META_KEY_N_EMBD: 256,  # Default for old SMG1
                C.META_KEY_N_HEAD: 8,
                C.META_KEY_N_HEAD_KV: 8,
                C.META_KEY_N_BLOCKS: 2,
                C.META_KEY_EPS: C.DEFAULT_EPS,
            }
            # Attempt to extract what we can from alternative metadata keys if they exist
            # (Some old models put everything in a single JSON)
            try:
                raw_meta = json.loads(
                    db_reader.read_metadata("config")
                )  # Re-try if just missing sub-keys
            except:
                raw_meta = {}

            meta.update(raw_meta)

        # Ensure critical keys exist with defaults
        config_data = meta.get(C.META_KEY_CONFIG, {})
        if "name" not in config_data:
            config_data["name"] = "legacy_recovered"

        valid_keys = set(ArchitectureConfig.__dataclass_fields__.keys())
        filtered_config = {k: v for k, v in config_data.items() if k in valid_keys}
        config = ArchitectureConfig(**filtered_config)

        # Instantiate model directly without `__init__` calling random generation
        model = cls.__new__(cls)
        model.config = config
        model.n_embd = meta.get(C.META_KEY_N_EMBD, 576)
        model.n_head = meta.get(C.META_KEY_N_HEAD, 9)
        model.n_head_kv = meta.get(C.META_KEY_N_HEAD_KV, 3)
        model.head_dim = model.n_embd // model.n_head if model.n_head > 0 else 64
        model.n_blocks = meta.get(C.META_KEY_N_BLOCKS, 30)
        model.eps = meta.get(C.META_KEY_EPS, C.DEFAULT_EPS)
        model.rope_base = config_data.get("rope_base", C.DEFAULT_ROPE_BASE)

        print(f"🧬 Despertando Organismo GAJE desde base de datos: {input_path}")
        start_total = time.time()

        def load_linear(name, out_features, in_features):
            # TENSOR NAME MAPPING LOGIC
            actual_name = name
            is_legacy_packed = False

            if not db_reader.has_tensor(f"{name}.dna"):
                # Try aliases from LEGACY_TENSOR_MAP
                aliases = C.LEGACY_TENSOR_MAP.get(name, [])
                for alias in aliases:
                    if db_reader.has_tensor(f"{alias}.dna") or db_reader.has_tensor(
                        alias
                    ):
                        actual_name = alias
                        if "packed_weights" in alias or not alias.endswith(".dna"):
                            is_legacy_packed = True
                        print(
                            f"🔗 Mapped tensor: '{name}' -> '{actual_name}' (Legacy Packed: {is_legacy_packed})"
                        )
                        break
                else:
                    if "blk." in name and not name.startswith("blk.0"):
                        print(f"⚠️ Skipping non-existent layer in small model: {name}")
                        return None

            # Loading Strategy
            if is_legacy_packed:
                # SMG1 or Legacy packed format
                dna = db_reader.read_tensor(actual_name)
                # For legacy packed, we might need to synthesize centroids if not present
                # Old SMG1 used fixed [-1.5, -0.5, 0.5, 1.5]
                centroids = meta.get("centroides", [-1.5, -0.5, 0.5, 1.5])
                # Expand centroids for all blocks
                n_blocks = (out_features * in_features) // 32
                centroids = centroids * n_blocks
                anchors_u8 = b""  # Legacy didn't have anchors
            else:
                dna = db_reader.read_tensor(f"{actual_name}.dna")
                centroids = np.frombuffer(
                    db_reader.read_tensor(f"{actual_name}.centroids"), dtype=np.float32
                ).tolist()
                anchors_u8 = db_reader.read_tensor(f"{actual_name}.anchors")

            bias = []
            if db_reader.has_tensor(f"{actual_name}.bias"):
                bias = np.frombuffer(
                    db_reader.read_tensor(f"{actual_name}.bias"), dtype=np.float32
                ).tolist()

            precision_mask = []
            epi_dna = b""
            epi_centroids = []
            tri_dna = b""
            tri_centroids = []

            if db_reader.has_tensor(f"{name}.precision_mask"):
                precision_mask = list(db_reader.read_tensor(f"{name}.precision_mask"))
                epi_dna = db_reader.read_tensor(f"{name}.epi_dna")
                epi_centroids = np.frombuffer(
                    db_reader.read_tensor(f"{name}.epi_centroids"), dtype=np.float32
                ).tolist()
                tri_dna = db_reader.read_tensor(f"{name}.tri_dna")
                tri_centroids = np.frombuffer(
                    db_reader.read_tensor(f"{name}.tri_centroids"), dtype=np.float32
                ).tolist()

            # Inferencia de bit_depth
            n_elements = out_features * in_features
            expected_2bit = (n_elements + 3) // 4
            expected_4bit = (n_elements + 1) // 2
            expected_32bit = n_elements * 4

            if len(dna) == expected_32bit:
                bit_depth = 32
            elif len(dna) == expected_4bit:
                bit_depth = 4
            elif len(dna) == expected_2bit:
                bit_depth = 2
            else:
                # Fallback agresivo para este test
                bit_depth = 32

            linear = dna_semantic_compression.GenomicLinear(
                dna,
                anchors_u8,
                centroids,
                out_features,
                in_features,
                32,
                bias=bias,
                precision_mask=precision_mask,
                epigenetic_database=epi_dna,
                epigenetic_centroids=epi_centroids,
                triplet_database=tri_dna,
                triplet_centroids=tri_centroids,
                bit_depth=bit_depth,
            )

            # Create a mock wrapper for Python interface
            class MockLayer:
                def __init__(self, lin, anchors_bin):
                    self.linear = lin
                    self.block_size = 32
                    self.anchors_f16_bytes = anchors_bin

                def forward(self, x):
                    return np.array(
                        self.linear.forward(x.tolist() if hasattr(x, "tolist") else x),
                        dtype=np.float32,
                    )

                def get_row(self, idx):
                    return np.array(self.linear.get_row(idx), dtype=np.float32)

            return MockLayer(linear, anchors_u8)

        vocab_size = meta.get("vocab_size")
        if vocab_size is None:
            # Detectar desde el tamaño del tensor DNA de embeddings
            embd_dna = db_reader.read_tensor("token_embd.dna")
            vocab_size = (len(embd_dna) * 4) // model.n_embd
            print(f"[*] Vocab size detectado automáticamente: {vocab_size}")

        model.embeddings = load_linear("token_embd", vocab_size, model.n_embd)
        model.lm_head = load_linear("lm_head", vocab_size, model.n_embd)

        output_norm = np.ones(model.n_embd).astype(np.float32).tolist()
        if db_reader.has_tensor("output_norm"):
            output_norm = np.frombuffer(
                db_reader.read_tensor("output_norm"), dtype=np.float32
            ).tolist()

        rust_blocks = []
        model.blocks = []
        actual_n_blocks = 0
        for i in range(model.n_blocks):
            p = f"blk.{i}."
            q_gen = load_linear(
                p + "attn_q", model.n_head * model.head_dim, model.n_embd
            )
            # SAFETY CHECK: If block components are missing, stop reconstruction
            if q_gen is None:
                print(
                    f"🛑 Stopping block reconstruction at index {i} (Incomplete or missing block)"
                )
                break

            k_gen = load_linear(
                p + "attn_k", model.n_head_kv * model.head_dim, model.n_embd
            )
            v_gen = load_linear(
                p + "attn_v", model.n_head_kv * model.head_dim, model.n_embd
            )
            w_o = load_linear(
                p + "attn_output", model.n_embd, model.n_head * model.head_dim
            )

            # Use metadata or heuristics to determine FFN size
            # For Qwen/SmolLM, FFN is usually hidden_dim * (8/3) or similar
            # We check the centroids size to be sure
            def get_out_features(name, in_features):
                actual_name = name
                if not db_reader.has_tensor(f"{name}.dna"):
                    aliases = C.LEGACY_TENSOR_MAP.get(name, [])
                    for alias in aliases:
                        if db_reader.has_tensor(f"{alias}.dna"):
                            actual_name = alias
                            break
                    else:
                        return model.n_embd * 4

                dna_bytes = db_reader.read_tensor(f"{actual_name}.dna")
                dna_len = len(dna_bytes)

                # Si no tiene centroides, es F32 (32-bit)
                if not db_reader.has_tensor(f"{actual_name}.centroids"):
                    return dna_len // (in_features * 4)

                centroids_bytes = db_reader.read_tensor(f"{actual_name}.centroids")
                centroids_len = len(centroids_bytes) // 4  # en floats
                if centroids_len == 0:
                    return dna_len // (in_features * 4)

                # Probamos 2-bit
                out_2bit = (dna_len * 4) // in_features
                expected_c_2bit = (out_2bit * in_features) // 8
                if expected_c_2bit == centroids_len:
                    return out_2bit

                # Probamos 4-bit
                out_4bit = (dna_len * 2) // in_features
                expected_c_4bit = (out_4bit * in_features) // 2
                if expected_c_4bit == centroids_len:
                    return out_4bit

                # Fallback seguro
                return (dna_len * 4) // in_features

            ffn_hidden = get_out_features(p + "ffn_gate", model.n_embd)

            gate_gen = load_linear(p + "ffn_gate", ffn_hidden, model.n_embd)
            up_gen = load_linear(p + "ffn_up", ffn_hidden, model.n_embd)
            w_down = load_linear(p + "ffn_down", model.n_embd, ffn_hidden)

            if not all([k_gen, v_gen, w_o, gate_gen, up_gen, w_down]):
                print(f"🛑 Block {i} components incomplete. Stopping.")
                break

            attn_norm_data = np.ones(model.n_embd).astype(np.float32).tolist()
            if db_reader.has_tensor(p + "attn_norm"):
                attn_norm_data = np.frombuffer(
                    db_reader.read_tensor(p + "attn_norm"), dtype=np.float32
                ).tolist()

            ffn_norm_data = np.ones(model.n_embd).astype(np.float32).tolist()
            if db_reader.has_tensor(p + "ffn_norm"):
                ffn_norm_data = np.frombuffer(
                    db_reader.read_tensor(p + "ffn_norm"), dtype=np.float32
                ).tolist()

            attn = dna_semantic_compression.GenomicAttention(
                model.n_head,
                model.n_head_kv,
                model.head_dim,
                attn_norm_data,
                model.eps,
                model.rope_base,
            )

            act_fn = model.config.ffn_act if model.config else "swiglu"
            use_gen_norm = model.config.use_genomic_norm if model.config else False

            rust_block = dna_semantic_compression.RustGenomicBlock(
                i,
                attn,
                q_gen.linear,
                k_gen.linear,
                v_gen.linear,
                w_o.linear,
                gate_gen.linear,
                up_gen.linear,
                w_down.linear,
                ffn_norm_data,
                model.eps,
                act_fn,
                use_gen_norm,
            )

            class MockBlock:
                def __init__(self, rb, q, k, v, o, gate, up, down):
                    self.rust_block = rb
                    self.attn_layer = type(
                        "obj", (object,), {"q_gen": q, "k_gen": k, "v_gen": v, "w_o": o}
                    )
                    self.gate_gen = gate
                    self.up_gen = up
                    self.w_down = down

            model.blocks.append(
                MockBlock(
                    rust_block, q_gen, k_gen, v_gen, w_o, gate_gen, up_gen, w_down
                )
            )
            rust_blocks.append(rust_block)
            actual_n_blocks += 1

        model.n_blocks = actual_n_blocks
        print(f"✅ Reconstructed {model.n_blocks} transformer blocks.")

        model.rust_llm = dna_semantic_compression.RustGenomicLLM(
            model.embeddings.linear,
            rust_blocks,
            output_norm,
            model.lm_head.linear,
            model.eps,
        )

        # Carga de Tokenizador Soberana (desde la BD si es posible)
        from transformers import AutoTokenizer, PreTrainedTokenizerFast

        try:
            if db_reader.has_metadata("tokenizer"):
                import tempfile
                import json

                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False
                ) as tmp:
                    tmp.write(db_reader.read_metadata("tokenizer"))
                    tmp_path = tmp.name

                # Cargar como un tokenizador rápido directamente desde el archivo JSON
                model.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tmp_path)
                os.unlink(tmp_path)
                print("[*] Tokenizador cargado desde la base de datos genómica.")
            else:
                model.tokenizer = AutoTokenizer.from_pretrained(
                    model.config.tokenizer_id
                )
        except Exception as e:
            print(
                f"[!] Aviso: Fallo al cargar tokenizador soberano, reintentando con ID: {e}"
            )
            model.tokenizer = AutoTokenizer.from_pretrained(model.config.tokenizer_id)

        end_total = time.time()
        print(
            f"[*] Reconstrucción desde BD finalizada en {end_total - start_total:.2f}s"
        )
        return model
