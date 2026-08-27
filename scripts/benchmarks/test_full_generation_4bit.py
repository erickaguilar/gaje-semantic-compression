import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.nn.stabilized import GenomicLLM

model_path = os.path.join(PROJECT_ROOT, "models", "production", "smollm2_4bit.gaje")
print(f"🧬 Probando Generación de Texto con Modelo 4-bit Uniforme: {model_path}...")

llm = GenomicLLM.load_genomic(model_path)
llm.rust_llm.set_k_wta_ratio(0.0)

prompts = [
    "The capital of France is",
    "Count from 1 to 5:",
]

for prompt in prompts:
    print(f"\nPrompt: '{prompt}'")
    print("Respuesta: ", end="", flush=True)
    generated = ""
    for tok in llm.generate(prompt, max_new_tokens=20, temperature=0.0):
        print(tok, end="", flush=True)
        generated += tok
    print()
