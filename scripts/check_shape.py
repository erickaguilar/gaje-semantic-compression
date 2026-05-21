import gguf
import sys

model_path = "/data/data/com.termux/files/home/models/gguf/smollm2-135m-f16.gguf"
reader = gguf.GGUFReader(model_path)
for t in reader.tensors:
    if "token_embd" in t.name or "attn_q" in t.name:
        print(f"Tensor: {t.name}, Shape: {t.shape}")
