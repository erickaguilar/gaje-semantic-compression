import gguf
import numpy as np
import os
import sys

def check():
    GGUF_PATH = "models/gguf/smollm2-135m-f16.gguf"
    reader = gguf.GGUFReader(GGUF_PATH)
    t = next(x for x in reader.tensors if x.name == "blk.0.attn_q.weight")
    print(f"Name: {t.name}")
    print(f"Shape: {t.shape}")
    print(f"Data size: {t.data.size}")
    print(f"Data type: {t.data.dtype}")

check()
