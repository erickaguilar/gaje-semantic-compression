from dataclasses import dataclass, field
from typing import Dict, Optional, List

@dataclass
class ArchitectureConfig:
    name: str
    tokenizer_id: str
    rope_base: float = 10000.0
    has_bias: bool = False
    rope_style: str = "split"  # "split" (Llama/Qwen) or "interleaved"
    unpermute_weights: bool = True
    ffn_act: str = "swiglu"
    tensor_name_mapping: Dict[str, str] = field(default_factory=dict)
    
    # Custom patches or fixes
    apply_smollm_rope_patch: bool = False

# Registry of known architectures
ARCHITECTURES: Dict[str, ArchitectureConfig] = {
    "llama": ArchitectureConfig(
        name="llama",
        tokenizer_id="HuggingFaceTB/SmolLM2-135M-Instruct", # Default for small llama-like
        rope_base=10000.0,
        has_bias=False,
        rope_style="split",
        unpermute_weights=True
    ),
    "qwen2": ArchitectureConfig(
        name="qwen2",
        tokenizer_id="Qwen/Qwen2-0.5B",
        rope_base=1000000.0, # Qwen2 usually uses 1M
        has_bias=True,
        rope_style="split",
        unpermute_weights=True
    ),
    "smollm": ArchitectureConfig(
        name="smollm",
        tokenizer_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        rope_base=10000.0,
        has_bias=False,
        rope_style="split",
        unpermute_weights=True,
        apply_smollm_rope_patch=True
    ),
    "gaje_native": ArchitectureConfig(
        name="gaje_native",
        tokenizer_id="Qwen/Qwen2-0.5B", # Placeholder or custom
        rope_base=10000.0,
        has_bias=False,
        rope_style="split",
        unpermute_weights=False, # Born genomic doesn't need unpermuting if designed right
    )
}

def get_config(arch_name: str) -> ArchitectureConfig:
    return ARCHITECTURES.get(arch_name.lower(), ARCHITECTURES["llama"])

def detect_arch(reader) -> str:
    """Detects architecture from GGUF reader."""
    if "general.architecture" in reader.fields:
        part = reader.fields["general.architecture"].parts[-1]
        if hasattr(part, 'tobytes'): 
            arch = part.tobytes().decode('utf-8').strip('\x00')
        elif isinstance(part, (list, bytes, bytearray)):
            if isinstance(part, (bytes, bytearray)):
                arch = part.decode('utf-8').strip('\x00')
            else:
                arch = "".join([chr(x) for x in part]).strip('\x00')
        else:
            arch = str(part).strip('\x00')
        return arch.lower()
    return "llama"
