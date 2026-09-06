#!/usr/bin/env python3
"""
🧬 GAJE HELIX — Colab Runner para ejecución desatendida vía Google Colab CLI.
Automatiza la preparación, compilación Rust con Vulkan, anclaje en VRAM y crianza en GPU (A100/T4).
"""
import os
import subprocess
import sys

def run(cmd):
    print(f"\n🧬 [GAJE Colab] >> {cmd}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(f"⚠️ Comando finalizó con código: {res.returncode}")
        sys.exit(res.returncode)

def main():
    print("=" * 70)
    print("🧬 GAJE HELIX — Pipeline de Crianza GPU en Google Colab Pro")
    print("=" * 70)

    # 1. Verificación de GPU NVIDIA
    run("nvidia-smi")

    # 2. Instalación de Vulkan y herramientas de enlace
    run("apt-get update -qq && apt-get install -y -qq libvulkan1 libvulkan-dev vulkan-tools mesa-vulkan-drivers build-essential")

    # 3. Preparación del repositorio en /content
    workdir = "/content/gaje-semantic-compression"
    if not os.path.exists(workdir):
        run(f"git clone -b main https://github.com/erickaguilar/gaje-semantic-compression.git {workdir}")
    os.chdir(workdir)
    run("git pull origin main")
    run("git submodule update --init --recursive")

    # 4. Instalación de Rust
    cargo_bin = os.path.expanduser("~/.cargo/bin/cargo")
    if not os.path.exists(cargo_bin):
        run("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")
    os.environ["PATH"] = f"{os.path.expanduser('~/.cargo/bin')}:{os.environ['PATH']}"
    run("rustc --version && cargo --version")

    # 5. Descarga de organismo max_512_pro.gaje si no está presente
    run("mkdir -p models/born")
    model_path = "models/born/max_512_pro.gaje"
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 100 * 1024 * 1024:
        run(f"curl -L -o {model_path} https://huggingface.co/eaguilar/gaje-models/resolve/main/max_512_pro.gaje")

    # 6. Compilación nativa en modo release con acelerador WGPU
    run("cargo build --release --bin gaje-cli")

    # 7. Validar tests de VRAM y shaders WGSL
    run("cargo test --test test_vram_viability -- --nocapture")
    run("cargo test --test test_gpu_integration -- --nocapture")

    # 8. Crianza del organismo en GPU
    epochs = sys.argv[1] if len(sys.argv) > 1 else "20"
    layers = sys.argv[2] if len(sys.argv) > 2 else "4"
    run(f"./target/release/gaje-cli crianza -m {model_path} -d data/genesis_conversational_corpus.jsonl -e {epochs} -l {layers} --gpu")

    # 9. Prueba de inferencia
    run(f'./target/release/gaje-cli --model {model_path} --prompt "¿Quién eres y qué puedes hacer?" --max-tokens 80')
    print("\n✅ ¡Entrenamiento en Colab GPU completado exitosamente!")

if __name__ == "__main__":
    main()
