#!/usr/bin/env bash
# =============================================================================
# GAJE HELIX — GPU Setup & Training Runner for Google Colab / Dedicated Servers
# =============================================================================
set -e

echo "🧬 [GAJE GPU] Verificando entorno de hardware..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "✅ GPU NVIDIA detectada."
else
    echo "ℹ️ Verificando adaptadores gráficos disponibles..."
    ls -l /dev/dri || true
fi

echo "📦 [GAJE GPU] Instalando bibliotecas nativas de Vulkan / WGPU..."
sudo apt-get update -qq
sudo apt-get install -y -qq libvulkan1 libvulkan-dev vulkan-tools mesa-vulkan-drivers build-essential

echo "🦀 [GAJE GPU] Verificando compilador Rust..."
if ! command -v cargo &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

echo "🚀 [GAJE GPU] Compilando núcleo GAJE nativo con acelerador GPU (Vulkan / WGPU)..."
cargo build --release --bin gaje-cli

echo "🧪 [GAJE GPU] Validando paridad e integración de pipelines WGSL..."
cargo test --test test_gpu_integration -- --nocapture

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Entorno GPU listo. Para iniciar crianza con Ladder Training en GPU:"
echo "   ./target/release/gaje-cli crianza -m models/born/max_512_pro.gaje -d data/genesis_conversational_corpus.jsonl -e 10 -l 4 --gpu"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
