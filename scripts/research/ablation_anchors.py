import os
import sys
import gc
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.nn.stabilized import GenomicLLM
from gaje.nn.configs import ARCHITECTURES

gguf_path = os.path.join(
    PROJECT_ROOT, "data", "models", "smollm2-135m-instruct-fp16.gguf"
)
model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
hf_model.eval()

prompt = "The capital of France is"
input_ids = tokenizer.encode(prompt, add_special_tokens=False)

inputs = torch.tensor([input_ids])
with torch.no_grad():
    hf_outputs = hf_model(inputs)
    hf_logits = hf_outputs.logits[0, -1, :].numpy()
    hf_top1 = int(np.argmax(hf_logits))

experiments = [
    ("Mixed-Bit + 10% FFN Anchors", 4, 2, 0.10),
    ("Mixed-Bit + 15% FFN Anchors", 4, 2, 0.15),
    ("Mixed-Bit + 20% FFN Anchors", 4, 2, 0.20),
]

print("\n======================================================")
print("🔬 BÚSQUEDA DE ANCLAJE OPTIMO FFN EN MIXED-BIT")
print("======================================================")

for name, attn_b, ffn_b, ffn_anchor in experiments:
    cfg = ARCHITECTURES["llama"]
    cfg.attn_bit_depth = attn_b
    cfg.ffn_bit_depth = ffn_b
    cfg.ffn_anchor_threshold = ffn_anchor

    out_file = os.path.join(
        PROJECT_ROOT, "models", f"test_{attn_b}b_{ffn_b}b_a{int(ffn_anchor * 100)}.gaje"
    )

    llm = GenomicLLM(gguf_path)
    gc.collect()
    llm.save(out_file)

    loaded = GenomicLLM.load_genomic(out_file)
    loaded.rust_llm.set_k_wta_ratio(0.0)

    g_logits = None
    for p_idx, tok_id in enumerate(input_ids):
        g_logits = np.array(loaded.rust_llm.forward(tok_id, p_idx == 0))

    cos = np.dot(hf_logits, g_logits) / (
        np.linalg.norm(hf_logits) * np.linalg.norm(g_logits) + 1e-9
    )
    top1 = int(np.argmax(g_logits))

    print(f"\n📊 Configuración: {name}")
    print(f"   - Similitud Coseno: {cos:.6f}")
    print(f"   - Predicción Top-1:  '{tokenizer.decode([top1])}' ({top1})")
    print(f"   - Coincidencia Top-1 vs HF: {'✅ SÍ' if top1 == hf_top1 else '❌ NO'}")

    del llm, loaded
    gc.collect()
