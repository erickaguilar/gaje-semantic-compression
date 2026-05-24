import os
import sys
import json
import time
import numpy as np

# Asegurar uso de código local
sys.path.insert(0, os.path.abspath("python"))

from gaje.nn.stabilized import GenomicLLM
from gaje.nn.configs import ArchitectureConfig
import gaje.core._impl as engine

def hatch_gold_embryo():
    print("🧬 GAJE: Incubando el Embrión de Oro (Phase 1)")
    print("=" * 60)
    
    # 1. Definir Configuración (SDD)
    config = ArchitectureConfig(
        name="gold_embryo",
        version="0.9.5-alpha",
        tokenizer_id="gpt2", # Usamos gpt2 como base por simplicidad de carga
        rope_base=10000.0,
        has_bias=False,
        rope_style="split",
        unpermute_weights=False,
        ffn_act="swiglu",
        use_genomic_norm=True # Crucial para modelos autonómicos
    )
    
    # 2. Inicializar el Organismo (Born-Genomic)
    # n_embd: 384, n_blocks: 8, n_head: 6, vocab: 16384
    # Usamos model_path=None para indicar que es un nacimiento desde cero
    print("[*] Instanciando arquitectura genómica (384, 8 blocks, 16k vocab)...")
    
    # Sobrescribimos el vocabulario para la prueba (normalmente viene del tokenizador)
    # En este caso, forzamos 16384 para cumplir el SDD.
    model = GenomicLLM(
        model_path=None, 
        num_blocks=8, 
        config=config
    )
    
    # Ajuste manual para cumplir el SDD exacto (ya que GenomicLLM podría usar defaults)
    model.n_embd = 384
    model.n_head = 6
    model.n_head_kv = 6
    model.head_dim = 64
    
    # 3. Guardar como .gaje
    output_path = "models/checkpoints/gold_embryo.gaje"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"[*] Guardando embrión en: {output_path}")
    model.save(output_path)
    
    print("=" * 60)
    print("✨ El Embrión de Oro ha nacido exitosamente.")

if __name__ == "__main__":
    hatch_gold_embryo()
