#!/usr/bin/env python3
import os
import subprocess
import sys

def run(cmd):
    print(f"\n🧬 [GAJE Colab] >> {cmd}", flush=True)
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(f"⚠️ Error {res.returncode}", flush=True)
        sys.exit(res.returncode)

def main():
    print("=" * 70, flush=True)
    print("🧬 GAJE HELIX — Pipeline de Crianza GPU en Google Colab Pro", flush=True)
    print("=" * 70, flush=True)

    # 1. GPU Check
    run("nvidia-smi")

    # 2. Vulkan
    run("apt-get update -qq && apt-get install -y -qq libvulkan1 libvulkan-dev vulkan-tools mesa-vulkan-drivers build-essential")

    # 3. Workdir
    workdir = "/content/gaje-semantic-compression"
    if not os.path.exists(workdir):
        run(f"git clone -b main https://github.com/erickaguilar/gaje-semantic-compression.git {workdir}")
    os.chdir(workdir)
    run("git pull origin main")
    run("git submodule update --init --recursive")

    # 4. Rust
    cargo_bin = os.path.expanduser("~/.cargo/bin/cargo")
    if not os.path.exists(cargo_bin):
        run("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")
    os.environ["PATH"] = f"{os.path.expanduser('~/.cargo/bin')}:{os.environ['PATH']}"
    run("rustc --version && cargo --version")

    # 5. Model
    run("mkdir -p models/born")
    model_path = "models/born/max_512_pro.gaje"
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 100 * 1024 * 1024:
        run(f"curl -L -o {model_path} https://huggingface.co/eaguilar/gaje-models/resolve/main/max_512_pro.gaje")

    # 6. Build
    run("cargo build --release --bin gaje-cli")

    # 7. Test VRAM
    run("cargo test --test test_vram_viability -- --nocapture")
    run("cargo test --test test_gpu_integration -- --nocapture")

    # 8. Train
    run(f"./target/release/gaje-cli crianza -m {model_path} -d data/genesis_conversational_corpus.jsonl -e 20 -l 4 --gpu")

    # 9. Inference
    run(f'./target/release/gaje-cli --model {model_path} --prompt "¿Quién eres y qué puedes hacer?" --max-tokens 80')
    print("\n✅ ¡Crianza en Colab GPU completada exitosamente!", flush=True)

if __name__ == "__main__":
    main()
