import numpy as np
import gguf
import os
import sys

sys.path.insert(0, os.path.abspath("python"))
from gaje.nn.stabilized import GenomicLLM

def scale_audit():
    gguf_path = "models/gguf/smollm2-135m-f16.gguf"
    gaje_path = "models/production/silver_adult_clean_v1.gaje"
    
    reader = gguf.GGUFReader(gguf_path)
    tensor = next(t for t in reader.tensors if t.name == "blk.0.attn_q.weight")
    w_original = np.frombuffer(tensor.data, dtype=np.float16).astype(np.float32)
    w_original = w_original.reshape(tensor.shape[::-1])
    
    llm = GenomicLLM.load_genomic(gaje_path)
    w_rec = np.zeros_like(w_original)
    for i in range(w_original.shape[0]):
        w_rec[i, :] = llm.blocks[0].attn_layer.q_gen.get_row(i)
        
    print(f"Original Max: {np.max(w_original):.4f}, Min: {np.min(w_original):.4f}, Mean Abs: {np.mean(np.abs(w_original)):.4f}")
    print(f"Reconstructed Max: {np.max(w_rec):.4f}, Min: {np.min(w_rec):.4f}, Mean Abs: {np.mean(np.abs(w_rec)):.4f}")

if __name__ == "__main__":
    scale_audit()
