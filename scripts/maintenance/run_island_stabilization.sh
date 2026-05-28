#!/bin/bash
# scripts/maintenance/run_island_stabilization.sh
set -e

BASE_MODEL="models/silver_adult_anchored.gaje"
NICHE_A="data/training/niche_A.txt"
NICHE_B="data/training/niche_B.txt"
NICHE_C="data/training/niche_C.txt"
OUTPUT_MODEL="models/silver_adult_stabilized.gaje"

# Asegurar que los binarios estén actualizados
echo "[*] Compilando herramientas nativas..."
cargo build --release --bin gaje-cli
cargo build --release --bin gaje-merger

# 1. Crear directorios para las islas
mkdir -p data/islands/

echo "--- Iniciando Fase 2: Entrenamiento por Nichos (Island Model) ---"

echo "[*] Entrenando Isla A (Sintaxis Base)..."
./target/release/gaje-cli $BASE_MODEL --train $NICHE_A --epochs 5 --save data/islands/island_A.gaje --resonance 0.1

echo "[*] Entrenando Isla B (Conectores y Fluidez)..."
./target/release/gaje-cli $BASE_MODEL --train $NICHE_B --epochs 5 --save data/islands/island_B.gaje --resonance 0.08

echo "[*] Entrenando Isla C (Morfología Verbal)..."
./target/release/gaje-cli $BASE_MODEL --train $NICHE_C --epochs 5 --save data/islands/island_C.gaje --resonance 0.05

echo "--- Iniciando Fase 3: Fusión por Promediado de Centroides ---"
./target/release/gaje-merger $OUTPUT_MODEL data/islands/island_A.gaje data/islands/island_B.gaje data/islands/island_C.gaje

echo "--- Fase 4: Validación de Generación ---"
echo "[*] Probando modelo estabilizado..."
./target/release/gaje-cli $OUTPUT_MODEL --prompt "GAJE es un protocolo"

echo ""
echo "[+] Proceso de estabilización completado exitosamente."
echo "[+] Modelo resultante: $OUTPUT_MODEL"
