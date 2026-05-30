"""Benchmark de precision con modelo GGUF local (adaptado para Windows)."""
import sys, os, numpy as np

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, os.path.join(project_root, "python"))
from gaje.nn.stabilized import GenomicLLM

model_path = os.path.join(project_root, "models", "SmolLM2-135M-Instruct-Q8_0.gguf")
phrases = [
    "2 + 2 =",
    "Paris is the capital of",
    "The sun rises in the",
    "Hello, my name is",
]

print("=" * 60)
print("  TEST DE PRECISION: BAJA ENTROPIA (GAJE 2-BIT)")
print("=" * 60)
print("[*] Cargando Organismo Genomico (4 bloques)...")
model = GenomicLLM(model_path, num_blocks=4)

for text in phrases:
    tokens = model.tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) < 1:
        continue
    logits = model.forward(tokens)[-1]
    probs = np.exp(logits - np.max(logits))
    probs /= probs.sum()
    top_indices = np.argsort(probs)[::-1][:5]
    top_probs = probs[top_indices]
    top_tokens = [model.tokenizer.decode([idx]) for idx in top_indices]
    print(f"\nContexto: '{text}'")
    print("   Top 5 predicciones:")
    for i in range(5):
        print(f'      {i+1}. "{top_tokens[i]}" ({top_probs[i]:.4f})')
print("=" * 60)
