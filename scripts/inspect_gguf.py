import gguf
import sys
path = "/data/data/com.termux/files/home/models/gguf/qwen2-0_5b-q8_0.gguf"
reader = gguf.GGUFReader(path)
print(f"--- METADATA KEYS ---")
for key in reader.fields.keys():
    print(key)
