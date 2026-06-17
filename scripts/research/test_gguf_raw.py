import sys
import gguf
import torch

def decode_gguf(gguf_path):
    print("Loading GGUF into torch...")
    # This is a bit complex, maybe just use ctransformers if available?
    pass

if __name__ == "__main__":
    decode_gguf("models/gguf/smollm2-135m-f16.gguf")
