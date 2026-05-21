import gguf
path = "/data/data/com.termux/files/home/models/gguf/qwen2-0_5b-q8_0.gguf"
reader = gguf.GGUFReader(path)
print(f"--- TENSOR NAMES ---")
for t in reader.tensors:
    print(t.name)
