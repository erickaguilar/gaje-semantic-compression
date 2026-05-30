from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ArchitectureConfig:
    name: str
    version: str = "0.9.5"
    tokenizer_id: str = "gpt2"
    rope_base: float = 10000.0
    has_bias: bool = False
    rope_style: str = "split"  # "split" (Llama/Qwen) or "interleaved"
    unpermute_weights: bool = True
    ffn_act: str = "swiglu"
    use_genomic_norm: bool = False
    tensor_name_mapping: Dict[str, str] = field(default_factory=dict)
    default_centroids: Dict[str, List[float]] = field(default_factory=dict)

    # Quantization Settings
    anchor_threshold: float = -1.0  # -1.0 means disable by default
    ffn_anchor_threshold: float = -1.0

    # Custom patches or fixes
    apply_smollm_rope_patch: bool = False
    dni: bool = False  # Direct Neural Ingestion support
    state: str = "stable"


# Registry of known architectures
ARCHITECTURES: Dict[str, ArchitectureConfig] = {
    "llama": ArchitectureConfig(
        name="llama",
        version="0.9.5",
        tokenizer_id="HuggingFaceTB/SmolLM2-135M-Instruct",  # Default for small llama-like
        rope_base=10000.0,
        has_bias=False,
        rope_style="split",
        unpermute_weights=True,
    ),
    "qwen2": ArchitectureConfig(
        name="qwen2",
        version="0.9.5",
        tokenizer_id="Qwen/Qwen2-0.5B",
        rope_base=1000000.0,  # Qwen2 usually uses 1M
        has_bias=True,
        rope_style="split",
        unpermute_weights=True,
        default_centroids={
            "blk.0.ffn_down.weight": [-0.0267, -0.0078, 0.0075, 0.0264],
            "blk.0.ffn_gate.weight": [-0.0364, -0.0132, 0.006, 0.0294],
            "blk.0.ffn_up.weight": [-0.0283, -0.0101, 0.0054, 0.0243],
            "blk.0.attn_q.weight": [-0.1034, -0.0199, 0.0268, 0.1148],
            "blk.1.ffn_down.weight": [-0.0253, -0.0082, 0.006, 0.0233],
            "blk.1.ffn_gate.weight": [-0.032, -0.0076, 0.0125, 0.0363],
            "blk.1.ffn_up.weight": [-0.0244, -0.0074, 0.0068, 0.0238],
            "blk.1.attn_q.weight": [-0.0482, -0.0133, 0.0091, 0.0428],
        },
    ),
    "smollm": ArchitectureConfig(
        name="smollm",
        version="0.9.5",
        tokenizer_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        rope_base=10000.0,
        has_bias=False,
        rope_style="split",
        unpermute_weights=True,
        apply_smollm_rope_patch=True,
    ),
    "gaje_native": ArchitectureConfig(
        name="gaje_native",
        version="0.9.5",
        tokenizer_id="Qwen/Qwen2-0.5B",  # Placeholder or custom
        rope_base=10000.0,
        has_bias=False,
        rope_style="split",
        unpermute_weights=False,
        ffn_act="relu",
        use_genomic_norm=True,
        anchor_threshold=0.15,
        ffn_anchor_threshold=0.15,
    ),
}


def get_config(arch_name: str) -> ArchitectureConfig:
    return ARCHITECTURES.get(arch_name.lower(), ARCHITECTURES["llama"])


def detect_arch(reader) -> str:
    """Detects architecture from GGUF reader."""
    if "general.architecture" in reader.fields:
        part = reader.fields["general.architecture"].parts[-1]
        if hasattr(part, "tobytes"):
            arch = part.tobytes().decode("utf-8").strip("\x00")
        elif isinstance(part, (list, bytes, bytearray)):
            if isinstance(part, (bytes, bytearray)):
                arch = part.decode("utf-8").strip("\x00")
            else:
                arch = "".join([chr(x) for x in part]).strip("\x00")
        else:
            arch = str(part).strip("\x00")
        return arch.lower()
    return "llama"
