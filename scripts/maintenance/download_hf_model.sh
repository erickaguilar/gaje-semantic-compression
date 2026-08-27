#!/usr/bin/env bash
# =============================================================================
# 🧬 GAJE — Script de Descarga Acelerada desde Hugging Face
# Soporta: hf_transfer (Rust), aria2c (16 hilos paralelos), curl y wget.
# =============================================================================

set -e

REPO_ID="${REPO_ID:-eaguilar/gaje-models}"
DEFAULT_OUT_DIR="./models/production"
OUT_DIR="${2:-$DEFAULT_OUT_DIR}"
MODEL_FILE="$1"

# Colores para salida terminal
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

mkdir -p "$OUT_DIR"

echo -e "${CYAN}=================================================================${NC}"
echo -e "${CYAN}🧬 GAJE — Descarga Acelerada de Modelos desde Hugging Face${NC}"
echo -e "${CYAN}=================================================================${NC}"

# Si no se pasó un archivo por parámetro, mostrar menú interactivo
if [ -z "$MODEL_FILE" ]; then
    echo -e "${YELLOW}Selecciona el modelo genómico a descargar:${NC}"
    echo "  1) gaje_pico_135m.flat    (~471 MB - Ideal para pruebas y WASM)"
    echo "  2) gaje_nano_1.5b.flat    (~650 MB - Portátiles y Edge)"
    echo "  3) gaje_prime_3b.flat     (~1.4 GB - Balance Productivo)"
    echo "  4) gaje_ultra_7b.flat     (~3.8 GB - Razonamiento Avanzado)"
    echo "  5) Ingresar nombre personalizado"
    echo ""
    read -p "Ingresa una opción [1-5]: " OPTION

    case "$OPTION" in
        1) MODEL_FILE="gaje_pico_135m.flat" ;;
        2) MODEL_FILE="gaje_nano_1.5b.flat" ;;
        3) MODEL_FILE="gaje_prime_3b.flat" ;;
        4) MODEL_FILE="gaje_ultra_7b.flat" ;;
        5) read -p "Nombre del archivo en Hugging Face: " MODEL_FILE ;;
        *) echo -e "${RED}Opción inválida.${NC}"; exit 1 ;;
    esac
fi

HF_URL="https://huggingface.co/${REPO_ID}/resolve/main/${MODEL_FILE}"
TARGET_PATH="${OUT_DIR}/${MODEL_FILE}"

echo -e "📦 Modelo objetivo: ${GREEN}${MODEL_FILE}${NC}"
echo -e "📂 Destino local:   ${GREEN}${TARGET_PATH}${NC}"
echo -e "🌐 Repositorio HF:  ${GREEN}${REPO_ID}${NC}"
echo ""

# Detección del método más rápido
if command -v aria2c &> /dev/null; then
    echo -e "${GREEN}⚡ Usando aria2c (16 conexiones paralelas concurrentes)...${NC}"
    aria2c -x 16 -s 16 -k 1M -c \
        --dir="$OUT_DIR" \
        --out="$MODEL_FILE" \
        "$HF_URL"

elif python3 -c "import hf_transfer, huggingface_hub" &> /dev/null; then
    echo -e "${GREEN}⚡ Usando hf_transfer (Motor nativo Rust Hugging Face)...${NC}"
    HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c "
import os
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id='${REPO_ID}', filename='${MODEL_FILE}', local_dir='${OUT_DIR}')
print('Descargado:', path)
"

elif command -v curl &> /dev/null; then
    echo -e "${YELLOW}⚡ Usando curl (Reanudación activada -C -)...${NC}"
    curl -L -C - --progress-bar -o "$TARGET_PATH" "$HF_URL"

elif command -v wget &> /dev/null; then
    echo -e "${YELLOW}⚡ Usando wget (Reanudación activada -c)...${NC}"
    wget -c -O "$TARGET_PATH" "$HF_URL"

else
    echo -e "${RED}❌ No se encontró ninguna herramienta de descarga (aria2c, python hf_transfer, curl, wget).${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ ¡Descarga completada con éxito!${NC}"
echo -e "Archivo guardado en: ${CYAN}${TARGET_PATH}${NC}"
ls -lh "$TARGET_PATH"
echo ""
echo -e "Para ejecutar inferencia inmediata con GAJE CLI:"
echo -e "  ${CYAN}cargo run --release -p gaje-cli -- chat --model \"${TARGET_PATH}\" --prompt \"Hola\"${NC}"
