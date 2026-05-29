#!/bin/bash

# 🧬 GAJE-Flow: Gestor de Ejecución Segura (Multi-OS: Android/Linux)
# Optimizado especialmente para Fedora Workstation y Termux.
# Este script asegura que el sistema no se suspenda ni entre en idle durante la destilación.

# 1. Configuración
BINARY="./target/release/micro-distiller"
LOG_FILE="benchmarks/logs/micro_distillation_$(date +%Y%m%d_%H%M%S).log"

# Colores para la interfaz
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Detectar el entorno y las capacidades del OS
USE_TERMUX=false
USE_SYSTEMD=false
USE_GNOME=false

if command -v termux-wake-lock &> /dev/null; then
    USE_TERMUX=true
fi

if command -v systemd-inhibit &> /dev/null; then
    USE_SYSTEMD=true
fi

if command -v gnome-session-inhibit &> /dev/null && [ -n "$DISPLAY" ]; then
    USE_GNOME=true
fi

# Función para liberar bloqueos de suspensión al salir
cleanup() {
    echo -e "\n${YELLOW}[*] Liberando bloqueos de energía y finalizando...${NC}"
    if [ "$USE_TERMUX" = true ]; then
        termux-wake-unlock
    fi
    exit
}

# Capturar señales de interrupción (Ctrl+C, etc.)
trap cleanup SIGINT SIGTERM

# 2. Información del Sistema (Optimizado para Fedora)
echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}🔬 DIAGNÓSTICO DE SISTEMA (Fedora/Linux Host)${NC}"
echo -e "${CYAN}============================================================${NC}"
if command -v lscpu &> /dev/null; then
    CPU_MODEL=$(lscpu | grep -E 'Model name|Nombre del modelo' | head -n 1 | awk -F: '{print $2}' | sed 's/^[ \t]*//')
    echo -e "   - CPU: ${GREEN}${CPU_MODEL}${NC}"
fi
echo -e "   - Núcleos disponibles: ${GREEN}$(nproc)${NC}"
if command -v free &> /dev/null; then
    MEM_FREE=$(free -h | awk '/Mem:/ {print $4 " libres / " $2 " totales"}')
    echo -e "   - Memoria RAM: ${GREEN}${MEM_FREE}${NC}"
fi
echo -e "   - Sistema Operativo: ${GREEN}$(uname -srm)${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

# 3. Verificaciones Previas y Compilación
echo -e "${YELLOW}[*] Compilando binario de destilación...${NC}"
cargo build --release --bin micro-distiller || exit 1

# 4. Inicio del Proceso con Ingestión de Wake Locks
clear
echo -e "${GREEN}------------------------------------------------------------${NC}"
echo -e "${GREEN}    GAJE-Flow: MODO DE CRIANZA INTENSIVA (MULTI-OS)         ${NC}"
echo -e "${GREEN}------------------------------------------------------------${NC}"
echo -e "[*] Log de salida: ${YELLOW}$LOG_FILE${NC}"

# Definir la ejecución final basada en el soporte de bloqueo de suspensión
EXEC_CMD="$BINARY"

if [ "$USE_TERMUX" = true ]; then
    echo -e "[*] Entorno Android/Termux detectado. Solicitando Wake Lock..."
    termux-wake-lock
elif [ "$USE_GNOME" = true ]; then
    echo -e "[*] Entorno Gráfico Fedora (GNOME) detectado. Usando gnome-session-inhibit..."
    EXEC_CMD="gnome-session-inhibit --app-id=\"gaje.micro.distiller\" --reason=\"Training DNA Semantic Compression\" --inhibit=idle --inhibit=suspend $BINARY"
elif [ "$USE_SYSTEMD" = true ]; then
    echo -e "[*] Entorno Linux (Systemd) detectado. Usando systemd-inhibit..."
    EXEC_CMD="systemd-inhibit --what=\"idle:sleep\" --who=\"GAJE Distiller\" --why=\"Training DNA Semantic Compression\" --mode=block $BINARY"
else
    echo -e "${YELLOW}[!] Advertencia: No se detectó termux-wake-lock ni systemd-inhibit.${NC}"
    echo -e "[*] Ejecutando directamente sin protección contra suspensión..."
fi

echo -e "------------------------------------------------------------"
# Asegurar la creación de la carpeta de logs
mkdir -p benchmarks/logs

# Ejecutar el comando final redirigiendo salida
eval "$EXEC_CMD" 2>&1 | tee "$LOG_FILE"

# 5. Finalización Exitosa
echo -e "------------------------------------------------------------"
echo -e "${GREEN}[✔] Proceso de destilación finalizado de forma segura.${NC}"
cleanup
