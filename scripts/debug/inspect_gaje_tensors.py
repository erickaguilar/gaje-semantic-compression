import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.core import _impl as C

fp32_path = os.path.join(PROJECT_ROOT, "models", "smollm2-135m-fp32.gaje")
db_reader = C.GajeDatabaseReader(fp32_path)

print("--- TENSORS IN .gaje DATABASE ---")
for key in [
    "output_norm",
    "token_embd.dna",
    "lm_head.dna",
    "blk.0.attn_norm",
    "blk.0.ffn_norm",
    "blk.29.attn_norm",
    "blk.29.ffn_norm",
]:
    if db_reader.has_tensor(key):
        print(f"✅ Found: '{key}'")
    else:
        print(f"❌ Missing: '{key}'")
