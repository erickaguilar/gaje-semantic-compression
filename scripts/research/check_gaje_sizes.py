import redb
import lz4_flex

db = redb.Database("models/production/silver_adult_clean_v1.gaje")
txn = db.begin_read()
table = txn.open_table("tensors")

keys = [
    "blk.0.attn_q.dna",
    "blk.0.attn_k.dna",
    "blk.0.attn_v.dna",
    "blk.0.attn_output.dna",
]

for k in keys:
    val = table.get(k)
    data = lz4_flex.decompress_size_prepended(val.value())
    print(f"{k}: len={len(data)}")
