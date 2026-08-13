import os
import sys
import gc

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.nn.stabilized import GenomicLLM
from gaje.nn.configs import ARCHITECTURES

gguf_path = os.path.join(
    PROJECT_ROOT, "data", "models", "smollm2-135m-instruct-fp16.gguf"
)
out_path = os.path.join(PROJECT_ROOT, "models", "production", "smollm2_4bit.gaje")

os.makedirs(os.path.dirname(out_path), exist_ok=True)

print(f"🧬 Generando modelo de producción 4-bit Uniforme: {out_path}...")
cfg = ARCHITECTURES["llama"]
cfg.attn_bit_depth = 4
cfg.ffn_bit_depth = 4
cfg.ffn_anchor_threshold = -1.0

llm = GenomicLLM(gguf_path)
gc.collect()
llm.save(out_path)
print(f"✅ Modelo de producción 4-bit guardado exitosamente en: {out_path}")
