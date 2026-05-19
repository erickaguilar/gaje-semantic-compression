import gguf
reader = gguf.GGUFReader("models/SmolLM2-135M-Instruct-Q8_0.gguf")
t = next(t for t in reader.tensors if "ffn_down" in t.name)
print(t.name, t.shape)
