#!/bin/bash

# 🧬 GAJE Validation Script - SmolLM2-v0.6.5
# Este script automatiza las pruebas sobre el organismo genómico persistido.

MODEL_PATH="experiments/smollm2_v065/smollm2-evolved.gaje"
CLI_PATH="./target/release/gaje-cli"
RESULTS_DIR="experiments/smollm2_v065/results"

echo "===================================================="
echo "🧪 Iniciando Validación de Organismo Genómico"
echo "Modelo: $MODEL_PATH"
echo "===================================================="

# 1. Prueba de Carga y Coherencia Básica
echo "[*] Prueba 1: Generación con prompt entrenado..."
$CLI_PATH $MODEL_PATH --prompt "Hola, soy un asistente inteligente." > $RESULTS_DIR/coherence_test.txt 2>&1
echo "[+] Resultado guardado en $RESULTS_DIR/coherence_test.txt"

# 2. Prueba de Creatividad (Zero-shot)
echo "[*] Prueba 2: Generación con prompt nuevo..."
$CLI_PATH $MODEL_PATH --prompt "¿Cuál es la capital de Francia?" > $RESULTS_DIR/zero_shot_test.txt 2>&1
echo "[+] Resultado guardado en $RESULTS_DIR/zero_shot_test.txt"

# 3. Reporte de Tamaño
echo "[*] Reporte de Densidad:"
ls -lh $MODEL_PATH | awk '{print "   > Tamaño del Cerebro Digital: " $5}'
echo "===================================================="
echo "✅ Validación finalizada."
