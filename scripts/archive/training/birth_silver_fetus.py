import os
import sys
import time

# Asegurar que usamos el código local de 'python/'
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python"))
)
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "python", "gaje", "core", "_impl")
    ),
)

from _impl import ArchConfig, ModelConfig, init_born_genomic_model, save_genomic_model


def birth_silver_fetus():
    print("🧬 Iniciando Nacimiento del Silver Fetus (v2.0 - 10MB) 🧬")
    print("-" * 60)

    # 1. Configuración del Organismo (Silver Fetus Specs)
    # n_embd=512, n_blocks=12, vocab=32768 => ~10MB (2-bit)
    name = "SilverFetus-v1"
    version = "0.9.8-alpha"
    vocab_size = 32768
    n_embd = 512
    n_blocks = 12
    tokenizer_path = "models/core/tokenizer.json"

    arch = ArchConfig(
        name=name,
        version=version,
        tokenizer_id=tokenizer_path,
        rope_base=1000000.0,
        ffn_act="swiglu",
        use_genomic_norm=True,
        rope_style="split",
    )

    config = ModelConfig(
        config=arch,
        n_embd=n_embd,
        n_head=8,
        n_head_kv=8,
        n_blocks=n_blocks,
        vocab_size=vocab_size,
        eps=1e-6,
    )

    model_dir = f"models/checkpoints/{name.lower()}"
    model_path = f"{model_dir}/model.gaje"
    os.makedirs(model_dir, exist_ok=True)

    # 2. Inicialización Nativa con Soberanía Algebraica
    # El motor de Rust cargará automáticamente models/core/algebraic_codebook.json
    print(f"[*] Creando arquitectura algebraica en {model_path}...")
    start_time = time.time()

    rust_llm = init_born_genomic_model(model_path, config, vocab_size)

    duration = time.time() - start_time
    print(f"✅ Organismo '{name}' inicializado en {duration:.2f}s.")
    print(f"[*] Estructura: {n_blocks} bloques, {n_embd} embd, {vocab_size} tokens.")

    # 3. Verificación de Guardado
    save_genomic_model(model_path, rust_llm, config, tokenizer_path)
    print("✨ Silver Fetus nacido exitosamente. Listo para Destilación CoT.")


if __name__ == "__main__":
    birth_silver_fetus()
