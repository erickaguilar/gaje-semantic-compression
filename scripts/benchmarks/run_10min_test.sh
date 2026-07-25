#!/bin/bash
# 🧪 Test de Crianza Nativa (10 Minutos)
# Este script ejecuta una destilación acotada para observar el flujo de memoria,
# el rendimiento de la CPU (ARM/Android) y la convergencia semántica (Loss/PPL).

set -e

echo "============================================================"
echo "🚀 INICIANDO TEST DE CRIANZA DE 10 MINUTOS (VÍA B)"
echo "============================================================"

# Definición de archivos
STUDENT="models/production/student_v1.gaje"
TEACHER="models/gguf/smollm2-135m-f16.gguf"
DATASET="data/datasets/10min_test_dataset.txt"

# 1. Monitoreo de Hardware (Opcional, en segundo plano)
echo "[*] Iniciando telemetría de hardware..."
if command -v top > /dev/null; then
    top -n 1 -b | head -n 5
fi

# 2. Ejecutar la micro-destilación cronometrada
echo "[*] Arrancando micro-distiller nativo..."
time ./target/release/micro-distiller \
    --student "$STUDENT" \
    --teacher "$TEACHER" \
    --tokenizer models/core/tokenizer.json \
    --dataset "$DATASET" \
    --epochs 5 \
    --lr 0.005 \
    --output "$STUDENT"

echo "============================================================"
echo "✅ TEST FINALIZADO"
echo "============================================================"
echo "Si la pérdida (Loss) disminuyó constantemente sin generar NaNs,"
echo "el motor y el flujo de destilación están listos para la Gran Destilación."
