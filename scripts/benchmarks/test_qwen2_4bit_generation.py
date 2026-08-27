import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.nn.stabilized import GenomicLLM

model_path = os.path.join(PROJECT_ROOT, "models", "production", "qwen2_0_5b_4bit.gaje")
print(f"🧬 Probando Generación con Qwen2-0.5B (4-bit Uniforme): {model_path}...")

llm = GenomicLLM.load_genomic(model_path)
llm.rust_llm.set_k_wta_ratio(0.0)

questions = [
    "¿Cuál es la capital de Francia?",
    "What is the largest planet in our solar system?",
]

for q in questions:
    formatted = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    print(f"\n--- PREGUNTA: '{q}' ---")
    print("Respuesta Qwen2 4-bit: ", end="", flush=True)
    for tok in llm.generate(formatted, max_new_tokens=30, temperature=0.0):
        if "<|im_end|>" in tok:
            break
        print(tok, end="", flush=True)
    print()
