import os
import sys
import gc

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.nn.stabilized import GenomicLLM
from gaje.nn.configs import ARCHITECTURES

gguf_path = os.path.join(
    PROJECT_ROOT, "data", "models", "qwen2-0_5b-instruct-fp16.gguf"
)
out_path = os.path.join(PROJECT_ROOT, "models", "production", "qwen2_0_5b_4bit.gaje")

os.makedirs(os.path.dirname(out_path), exist_ok=True)

print(f"🧬 Generando modelo de producción Qwen2-0.5B 4-bit Uniforme: {out_path}...")
cfg = ARCHITECTURES["qwen2"]
cfg.attn_bit_depth = 4
cfg.ffn_bit_depth = 4
cfg.ffn_anchor_threshold = -1.0

llm = GenomicLLM(gguf_path)
gc.collect()
llm.save(out_path)

# Copiar también a la raíz de models/
import shutil

shutil.copy(out_path, os.path.join(PROJECT_ROOT, "models", "qwen2_0_5b_4bit.gaje"))

print(f"✅ Modelo de producción Qwen2-0.5B 4-bit guardado exitosamente en: {out_path}")
