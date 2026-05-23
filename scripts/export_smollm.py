import os
import sys
import numpy as np
import json
import gaje.core._impl as engine
from gaje.nn.stabilized import GenomicLLM

def export_smollm(gguf_path, output_path):
    print(f"🧬 Exporting SmolLM-135M to GAJE format: {gguf_path}")
    
    # 1. Genomize the model using the established protocol
    model = GenomicLLM(gguf_path)
    
    # 2. Create the Database Writer
    if os.path.exists(output_path):
        os.remove(output_path)
    
    writer = engine.GajeDatabaseWriter(output_path)
    batch = writer.begin_batch()
    
    def add_layer_packed(layer, name):
        # We store the packed weights (2-bit DNA)
        batch.write_tensor(f"{name}.dna", bytes(layer.database))
        
        # Centroids are essential
        batch.write_tensor(f"{name}.centroids", np.array(layer.centroids, dtype=np.float32).tobytes())
        
        # Anchors (Sparse buffer)
        batch.write_tensor(f"{name}.anchors", bytes(layer.anchors_raw))
        
        if hasattr(layer, 'bias') and layer.bias is not None and len(layer.bias) > 0:
            batch.write_tensor(f"{name}.bias", np.array(layer.bias, dtype=np.float32).tobytes())

        if hasattr(layer, 'precision_mask') and len(layer.precision_mask) > 0:
            batch.write_tensor(f"{name}.precision_mask", bytes(layer.precision_mask))
            batch.write_tensor(f"{name}.epi_dna", bytes(layer.epigenetic_database))
            batch.write_tensor(f"{name}.epi_centroids", np.array(layer.epigenetic_centroids, dtype=np.float32).tobytes())
            batch.write_tensor(f"{name}.tri_dna", bytes(layer.triplet_database))
            batch.write_tensor(f"{name}.tri_centroids", np.array(layer.triplet_centroids, dtype=np.float32).tobytes())

    print("[*] Packing layers into database...")
    # Embeddings
    add_layer_packed(model.rust_llm.embeddings, "token_embd")
    
    # Blocks
    for i, block in enumerate(model.rust_llm.blocks):
        prefix = f"blk.{i}."
        add_layer_packed(block.q_gen, prefix + "attn_q")
        add_layer_packed(block.k_gen, prefix + "attn_k")
        add_layer_packed(block.v_gen, prefix + "attn_v")
        add_layer_packed(block.w_o, prefix + "attn_output")
        add_layer_packed(block.gate_gen, prefix + "ffn_gate")
        add_layer_packed(block.up_gen, prefix + "ffn_up")
        add_layer_packed(block.w_down, prefix + "ffn_down")
        
        # Norm weights
        batch.write_tensor(prefix + "ffn_norm", np.array(block.ffn_norm, dtype=np.float32).tobytes())
        batch.write_tensor(prefix + "attn_norm", np.array(block.attn.rmsnorm_weight, dtype=np.float32).tobytes())

    # LM Head
    add_layer_packed(model.rust_llm.lm_head, "lm_head")
    batch.write_tensor("output_norm", np.array(model.rust_llm.output_norm, dtype=np.float32).tobytes())

    # 3. Save metadata
    config_meta = {
        "n_embd": model.n_embd,
        "n_head": model.n_head,
        "n_head_kv": model.n_head_kv,
        "n_blocks": model.n_blocks,
        "vocab_size": model.rust_llm.embeddings.out_features,
        "rope_base": model.rope_base,
        "eps": model.eps,
        "config": {
            "name": "llama",
            "rope_base": model.rope_base,
            "rope_style": "split",
            "ffn_act": "swiglu"
        }
    }
    batch.write_metadata("config", json.dumps(config_meta))
    
    # Commit changes
    batch.commit()
    
    print(f"✅ Export complete: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python export_smollm.py <gguf_path> <output_gaje_path>")
    else:
        export_smollm(sys.argv[1], sys.argv[2])
