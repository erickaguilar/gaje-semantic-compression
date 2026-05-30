"""
Centralized constants and default values for GAJE-Flow Protocol.
Ensures stability across different model versions and architectures.
"""

# Default Metadata Values
DEFAULT_VERSION = "0.9.5-alpha"
DEFAULT_ROPE_BASE = 10000.0
DEFAULT_EPS = 1e-5
DEFAULT_TOKENIZER_ID = "gpt2"
DEFAULT_FFN_ACT = "swiglu"
DEFAULT_BLOCK_SIZE = 32

# Metadata Keys
META_KEY_CONFIG = "config"
META_KEY_N_EMBD = "n_embd"
META_KEY_N_HEAD = "n_head"
META_KEY_N_HEAD_KV = "n_head_kv"
META_KEY_N_BLOCKS = "n_blocks"
META_KEY_EPS = "eps"
META_KEY_VOCAB_SIZE = "vocab_size"

# Model Signatures / Magic Strings
SIGNATURE_SMG1 = "smg1_spiking"
SIGNATURE_LLAMA = "llama"
SIGNATURE_QWEN = "qwen2"

# Legacy Mapping (Old keys -> New Config)
LEGACY_MAPPING = {
    "dim_latent": "n_embd",
    "dim_logic": "n_head",  # Rough approximation for legacy
    "type": "name",
}

# Tensor Name Mapping (Modern Name -> Old Name Aliases)
# This allows the loader to find "token_embd" even if it's called "l0" in the DB.
LEGACY_TENSOR_MAP = {
    "token_embd": ["layer.0.packed_weights", "l0", "embeddings", "model.embed_tokens"],
    "lm_head": ["layer.2.packed_weights", "l2", "output", "model.lm_head"],
    "blk.0.attn_q": ["layer.1.packed_weights", "l1.q", "layers.0.attention.wq"],
    "blk.0.attn_k": ["layer.1.packed_weights", "l1.k", "layers.0.attention.wk"],
    "blk.0.attn_v": ["layer.1.packed_weights", "l1.v", "layers.0.attention.wv"],
    "blk.0.attn_output": ["layer.1.packed_weights", "l1.o", "layers.0.attention.wo"],
    "blk.0.ffn_gate": ["layer.1.packed_weights", "l1.gate"],
    "blk.0.ffn_up": ["layer.1.packed_weights", "l1.up"],
    "blk.0.ffn_down": ["layer.1.packed_weights", "l1.down"],
}
