#!/bin/bash

# 🧬 GAJE-Flow: Gestor de Ejecución Segura (Wake Lock)
# Este script asegura que Android no suspenda la CPU durante la destilación.

# 1. Configuración
BINARY="./target/release/micro-distiller"
LOG_FILE="benchmarks/logs/micro_distillation_$(date +%Y%m%d_%H%M%S).log"

# Colores para la interfaz
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Función para liberar el Wake Lock al salir
cleanup() {
    echo -e "\n${YELLOW}[*] Liberando Wake Lock y finalizando...${NC}"
    termux-wake-unlock
    exit
}

# Capturar señales de interrupción (Ctrl+C, etc.)
trap cleanup SIGINT SIGTERM

# 2. Verificaciones Previas
if [ ! -f "$BINARY" ]; then
    echo -e "${YELLOW}[*] El binario no existe. Compilando...${NC}"
    cargo build --release --bin micro-distiller
fi

# Verificar si termux-api está instalado
if ! command -v termux-wake-lock &> /dev/null; then
    echo -e "${RED}[!] Error: termux-api no está instalado.${NC}"
    echo -e "Ejecuta: pkg install termux-api"
    exit 1
fi

# 3. Inicio del Proceso
clear
echo -e "${GREEN}------------------------------------------------------------${NC}"
echo -e "${GREEN}    GAJE-Flow: MODO DE CRIANZA INTENSIVA (WAKE LOCK)        ${NC}"
echo -e "${GREEN}------------------------------------------------------------${NC}"
echo -e "[*] Solicitando Wake Lock de Android..."
termux-wake-lock

echo -e "[*] Iniciando destilación micro..."
echo -e "[*] Log: ${YELLOW}$LOG_FILE${NC}"
echo -e "------------------------------------------------------------"

# Ejecutar el binario y redirigir salida a la vez que se muestra en pantalla (tee)
$BINARY 2>&1 | tee "$LOG_FILE"

# 4. Finalización Exitosa
echo -e "------------------------------------------------------------"
echo -e "${GREEN}[✔] Proceso finalizado.${NC}"
cleanup
