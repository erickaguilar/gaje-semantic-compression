import gguf
import sys

model_path = "/data/data/com.termux/files/home/models/gguf/smollm2-135m-f16.gguf"
reader = gguf.GGUFReader(model_path)

print(f"--- Metadata for {model_path} ---")
for key, field in reader.fields.items():
    if "architecture" in key or "rope" in key or "head_count" in key:
        print(f"{key}: {field.parts[-1]}")

print("\n--- Tensors (first 10) ---")
for i, tensor in enumerate(reader.tensors):
    if i < 10:
        print(f"{tensor.name}: {tensor.shape} ({tensor.tensor_type})")
