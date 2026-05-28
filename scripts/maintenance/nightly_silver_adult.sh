#!/bin/bash
# scripts/maintenance/nightly_silver_adult.sh
set -e

# Configuración
NEW_MODEL="models/silver_adult_nightly.gaje"
NICHE_A="data/training/niche_A.txt"
NICHE_B="data/training/niche_B.txt"
NICHE_C="data/training/niche_C.txt"
EPOCHS=100

echo "🧬 Iniciando Construcción Nocturna de Silver Adult (v1.0.0)"
echo "[*] Fecha: $(date)"

# 1. Inicializar nuevo organismo Silver Adult
echo "[*] Inicializando nuevo organismo con preset 'silver_adult'..."
./target/release/gaje-cli models/temp_init.gaje --init $NEW_MODEL --preset silver_adult

# 2. Entrenamiento por Islas (Evolución Paralela)
mkdir -p data/islands/nightly/

echo "--- Fase 2: Entrenamiento Intensivo (Island Model) ---"

echo "[*] Entrenando Isla A (Sintaxis Base) - $EPOCHS épocas..."
./target/release/gaje-cli $NEW_MODEL --train $NICHE_A --epochs $EPOCHS --save data/islands/nightly/island_A.gaje --resonance 0.12 --scale 0.01

echo "[*] Entrenando Isla B (Conectores y Fluidez) - $EPOCHS épocas..."
./target/release/gaje-cli $NEW_MODEL --train $NICHE_B --epochs $EPOCHS --save data/islands/nightly/island_B.gaje --resonance 0.10 --scale 0.01

echo "[*] Entrenando Isla C (Morfología Verbal) - $EPOCHS épocas..."
./target/release/gaje-cli $NEW_MODEL --train $NICHE_C --epochs $EPOCHS --save data/islands/nightly/island_C.gaje --resonance 0.08 --scale 0.01

echo "--- Fase 3: Fusión por Promediado de Centroides ---"
./target/release/gaje-merger models/silver_adult_nightly_merged.gaje data/islands/nightly/island_A.gaje data/islands/nightly/island_B.gaje data/islands/nightly/island_C.gaje

echo "--- Fase 4: Refinamiento Global Final ---"
# Entrenamos el modelo fusionado un poco más con el dataset completo para suavizar la integración
./target/release/gaje-cli models/silver_adult_nightly_merged.gaje --train data/datasets/full_silver_adult_dataset.txt --epochs 10 --save models/silver_adult_final_nightly.gaje --resonance 0.1 --scale 0.005

echo "[+] ¡Construcción Nocturna Finalizada!"
echo "[+] Modelo final: models/silver_adult_final_nightly.gaje"
date
