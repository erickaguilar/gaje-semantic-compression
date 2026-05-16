import gguf
reader = gguf.GGUFReader("/data/data/com.termux/files/home/models/smollm2-135m-f16.gguf")
t = next(t for t in reader.tensors if "ffn_down" in t.name)
print(t.name, t.shape)
