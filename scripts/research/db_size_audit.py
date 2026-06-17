import redb
import os
import lz4_flex
import struct

def audit_database_v2(path):
    print(f"Auditing DB: {path}")
    db = redb.Database(path)
    read_txn = db.begin_read()
    table = read_txn.open_table("tensors")
    
    # Check config
    config_val = read_txn.open_table("metadata").get("config")
    import json
    config = json.loads(config_val.value())
    n_embd = config["n_embd"]
    print(f"Config: n_embd={n_embd}, n_blocks={config['n_blocks']}")

    keys = ["blk.0.attn_v.dna", "blk.0.ffn_gate.dna"]
    for key in keys:
        val = table.get(key)
        if val:
            data = lz4_flex.decompress_size_prepended(val.value())
            size = len(data)
            # attn_v is (n_embd, n_embd) usually
            # gate is (n_embd, ffn_h)
            expected_2bit = (n_embd * n_embd) // 4
            expected_4bit = (n_embd * n_embd) // 2
            print(f"Tensor {key}: size={size} bytes")
            print(f"  - Expected 2-bit: {expected_2bit}")
            print(f"  - Expected 4-bit: {expected_4bit}")
            if size == expected_2bit: print("  -> Detected 2-bit")
            elif size == expected_4bit: print("  -> Detected 4-bit")
            else: print("  -> UNKNOWN DEPTH")

if __name__ == "__main__":
    audit_database_v2("models/production/silver_adult_sovereign.gaje")
    print("-" * 20)
    audit_database_v2("models/production/genesis_sovereign.gaje")
