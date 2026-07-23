import os
import sys
import numpy as np
from tqdm import tqdm
import gguf
import json

# Asegurar uso de código local
sys.path.insert(0, os.path.abspath("python"))

import gaje.core._impl as dna_semantic_compression
from gaje.nn import constants as C
from gaje.utils.quantization import unpermute_to_interleaved

def log_ram(step=""):
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if "VmRSS" in line:
                    mem = line.split(":")[1].strip()
                    print(f"📈 [RAM] {step}: {mem}")
    except:
        pass

def run_ptq_pipeline():
    GGUF_PATH = "models/gguf/smollm2-135m-f16.gguf"
    OUTPUT_PATH = "models/production/smollm2_mixed_v1.gaje"
    
    print(f"🚀 Iniciando Exportación Final PTQ Mixed-Bit: SmolLM2-135M")
    log_ram("Inicio")

    if not os.path.exists(GGUF_PATH):
        print(f"❌ Error: No se encuentra {GGUF_PATH}")
        return

    # Asegurar directorio de salida
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    reader = gguf.GGUFReader(GGUF_PATH)
    
    # 1. Mapear Tensores
    layers = {}
    other_tensors = {}
    for tensor in reader.tensors:
        name = tensor.name
        # Genomizamos solo pesos lineales reales. Los NORM deben ser directos.
        if "weight" in name and (".attn_q" in name or ".attn_k" in name or ".attn_v" in name or ".attn_output" in name or ".ffn_" in name or "token_embd" in name or "lm_head" in name):
            layers[name] = tensor
        else:
            other_tensors[name] = tensor

    print(f"✅ {len(layers)} capas para genomizar, {len(other_tensors)} tensores directos.")
    
    # 2. Inicializar Writer
    print(f"[Debug] GajeDatabaseWriter attrs: {dir(dna_semantic_compression.GajeDatabaseWriter)}")
    writer = dna_semantic_compression.GajeDatabaseWriter(OUTPUT_PATH)
    batch = writer.begin_batch()

    # Metadatos del modelo
    config = {
        "config": {
            "name": "SmolLM2-135M-Mixed-v1",
            "rope_style": "interleaved",
            "unpermute_weights": True
        },
        "n_embd": 576,
        "n_head": 9,
        "n_head_kv": 3,
        "n_blocks": 30,
        "vocab_size": 49152,
        "eps": 1e-5,
    }
    batch.write_metadata(C.META_KEY_CONFIG, json.dumps(config))
    
    # Escribir Tokenizador (formato compatible con la librería 'tokenizers' de Rust)
    with open("temp_tokenizer/tokenizer.json", "r") as f:
        tok_json = f.read()
    batch.write_metadata("tokenizer", tok_json)

    # 3. Proceso de Genomización
    print("[~] Genomizando y Empaquetando...")
    
    for name, tensor in tqdm(layers.items()):
        # Convertir a f32 y FLATTEN (Sin unpermute)
        w_f32 = tensor.data.astype(np.float32).flatten()
        
        # Mixed-Bit Logic: 4-bit para Atención y Capas Críticas, 2-bit para MLP
        is_critical = ".attn_" in name or "token_embd" in name or "lm_head" in name
        bit_depth = 4 if is_critical else 2
        block_size = 32
        anchor_rate = 0.01 # 1% anclas por magnitud (Zero-Forward approach)
        
        # Nombre base para la BD (remover .weight)
        base_name = name.replace(".weight", "")

        # Genomización Nativa (Rust)
        dna_bytes, centroids, anchors_buf = dna_semantic_compression.genomize_f32_native(
            w_f32.tobytes(),
            block_size,
            anchor_rate,
            bit_depth
        )
        
        # Escritura comprimida (LZ4 interna de Rust)
        batch.write_tensor_compressed(f"{base_name}.dna", dna_bytes)
        batch.write_tensor(f"{base_name}.centroids", np.array(centroids, dtype=np.float32).tobytes())
        batch.write_tensor(f"{base_name}.anchors", anchors_buf)
        
        # SI ES TOKEN_EMBD, DUPLICAMOS COMO LM_HEAD (Tied Weights support)
        if base_name == "token_embd":
            print("  [*] Duplicando token_embd como lm_head (Tied Weights)")
            batch.write_tensor_compressed("lm_head.dna", dna_bytes)
            batch.write_tensor("lm_head.centroids", np.array(centroids, dtype=np.float32).tobytes())
            batch.write_tensor("lm_head.anchors", anchors_buf)
        
    # 4. Tensores No-Genómicos
    print("[~] Escribiendo tensores de soporte...")
    for name, tensor in other_tensors.items():
        # Los embeddings y normas se escriben comprimidos para ahorrar espacio
        # REMOVER .weight para compatibilidad con loader.rs
        clean_name = name.replace(".weight", "")
        data = tensor.data.astype(np.float32).tobytes()
        batch.write_tensor_compressed(clean_name, data)

    # 5. Commit Final
    print("[~] Committing a la base de datos RedB...")
    batch.commit()
    
    print("-" * 50)
    print(f"✨ ÉXITO: Organismo generado en {OUTPUT_PATH}")
    file_size = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"📦 Tamaño final: {file_size:.2f} MB")
    log_ram("Fin")

if __name__ == "__main__":
    run_ptq_pipeline()
