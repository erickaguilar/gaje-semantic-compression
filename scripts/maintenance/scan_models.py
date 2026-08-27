import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
models_root = os.path.join(PROJECT_ROOT, "models")

print("--- SCANNING MODELS DIRECTORY RECURSIVELY ---")
seen = set()
for root, _, files in os.walk(models_root):
    for f in files:
        if f.endswith(".gaje") and f not in seen:
            print(f"✅ Found model: {f} (in {root})")
            seen.add(f)
