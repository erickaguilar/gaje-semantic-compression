import os
import sys
import numpy as np
from gaje.nn.stabilized import GenomicLLM
from gaje.core.archive import GAJEArchive
import json

def export_smollm(gguf_path, output_path):
    print(f"🧬 Exporting SmolLM-135M to GAJE format: {gguf_path}")
    
    # 1. Genomize the model using the established protocol
    # Note: We use num_blocks=None to load the full 30 layers
    model = GenomicLLM(gguf_path)
    
    # 2. Create the Archive
    # We'll use shared centroids if possible to save space
    shared_centroids = model.rust_llm.embeddings.centroids
    archive = GAJEArchive(codebook={"centroids": shared_centroids})
    
    def add_layer(layer, name):
        # We store the primary DNA strand. 
        # For a truly minimal SmolLM, we avoid epi/tri strands in this first pass.
        dna = bytes(layer.database)
        archive.add(f"{name}.dna", dna)
        # If the layer has unique centroids, we'd need to extend the format or store them as entries
        # For now, we assume shared centroids as per the 'codebook' concept
        if hasattr(layer, 'bias') and len(layer.bias) > 0:
            archive.add(f"{name}.bias", bytes(np.array(layer.bias, dtype=np.float32)))

    print("[*] Packing layers into archive...")
    # Embeddings
    add_layer(model.rust_llm.embeddings, "token_embd")
    
    # Blocks
    for i, block in enumerate(model.rust_llm.blocks):
        prefix = f"blk.{i}."
        add_layer(block.q_gen, prefix + "attn_q")
        add_layer(block.k_gen, prefix + "attn_k")
        add_layer(block.v_gen, prefix + "attn_v")
        add_layer(block.w_o, prefix + "attn_output")
        add_layer(block.gate_gen, prefix + "ffn_gate")
        add_layer(block.up_gen, prefix + "ffn_up")
        add_layer(block.w_down, prefix + "ffn_down")
        
        # Norm weights are small, we could store them as JSON in metadata or as raw entries
        archive.add(prefix + "ffn_norm", bytes(np.array(block.ffn_norm, dtype=np.float32)))
        # Attn norm is in GenomicAttention, we need to extract it
        archive.add(prefix + "attn_norm", bytes(np.array(block.attn.rmsnorm_weight, dtype=np.float32)))

    # LM Head
    add_layer(model.rust_llm.lm_head, "lm_head")
    archive.add("output_norm", bytes(np.array(model.rust_llm.output_norm, dtype=np.float32)))

    # 3. Save the Archive
    archive.save(output_path)
    
    # 4. Save metadata separately or integrated
    metadata = {
        "n_embd": model.n_embd,
        "n_head": model.n_head,
        "n_head_kv": model.n_head_kv,
        "n_blocks": model.n_blocks,
        "vocab_size": model.rust_llm.embeddings.out_features,
        "rope_base": model.rope_base,
        "eps": model.eps
    }
    with open(output_path + ".json", "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"✅ Export complete: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python export_smollm.py <gguf_path> <output_gaje_path>")
    else:
        export_smollm(sys.argv[1], sys.argv[2])
