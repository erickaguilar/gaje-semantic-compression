#!/bin/bash

# 🧬 GAJE-Flow: Script de Descarga/Restauración desde Google Drive (rclone)
# Este script descarga carpetas críticas (modelos, datasets) desde tu Google Drive.

# --- CONFIGURACIÓN ---
REMOTE_NAME="gdrive"           # Nombre que le diste en 'rclone config'
REMOTE_PATH="GAJE_Backup"      # Carpeta origen en Drive
LOG_FILE="benchmarks/logs/download_history.log"
DATE=$(date '+%Y-%m-%d %H:%M:%S')

# Lista de carpetas a descargar/restaurar
FOLDERS=(
    "docs"
    "models"
    "data/datasets"
    "benchmarks/logs"
)

# --- INICIO DEL PROCESO ---
mkdir -p benchmarks/logs

echo "------------------------------------------------------------" | tee -a "$LOG_FILE"
echo "[*] Iniciando Descarga de Archivos GAJE: $DATE" | tee -a "$LOG_FILE"
echo "------------------------------------------------------------" | tee -a "$LOG_FILE"

# Verificar si rclone está configurado
if ! rclone listremotes | grep -q "^$REMOTE_NAME:$"; then
    echo "[!] Error: El remoto '$REMOTE_NAME' no está configurado en rclone." | tee -a "$LOG_FILE"
    echo "[*] Por favor ejecuta 'rclone config' para configurar tu cuenta de Google Drive."
    echo "[*] Asegúrate de nombrar el remoto exactamente como: $REMOTE_NAME"
    exit 1
fi

for FOLDER in "${FOLDERS[@]}"; do
    echo "[>] Descargando/Sincronizando desde Drive: $FOLDER..." | tee -a "$LOG_FILE"

    # Crear carpeta local si no existe
    mkdir -p "$FOLDER"

    # rclone sync del remoto al local
    rclone sync "$REMOTE_NAME:$REMOTE_PATH/$FOLDER" "$FOLDER" \
        --progress \
        --drive-chunk-size 64M \
        --log-file="$LOG_FILE" \
        --log-level INFO

    if [ $? -eq 0 ]; then
        echo "[✔] $FOLDER descargado y sincronizado con éxito." | tee -a "$LOG_FILE"
    else
        echo "[X] Error al descargar $FOLDER. Revisa los logs en $LOG_FILE." | tee -a "$LOG_FILE"
    fi
done

echo "------------------------------------------------------------" | tee -a "$LOG_FILE"
echo "[*] Descarga finalizada a las $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"
echo "------------------------------------------------------------" | tee -a "$LOG_FILE"
