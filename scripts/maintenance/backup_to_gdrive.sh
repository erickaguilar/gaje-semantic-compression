#!/bin/bash

# 🧬 GAJE-Flow: Script de Respaldo Automatizado (rclone)
# Este script sincroniza carpetas críticas con Google Drive.

# --- CONFIGURACIÓN ---
REMOTE_NAME="gdrive"           # Nombre que le diste en 'rclone config'
REMOTE_PATH="GAJE_Backup"      # Carpeta destino en Drive
LOG_FILE="benchmarks/logs/backup_history.log"
DATE=$(date '+%Y-%m-%d %H:%M:%S')

# Lista de carpetas a respaldar (añade o quita según necesites)
FOLDERS=(
    "docs"
    "models"
    "data/datasets"
    "benchmarks/logs"
)

# --- INICIO DEL PROCESO ---
echo "------------------------------------------------------------" | tee -a "$LOG_FILE"
echo "[*] Iniciando Respaldo GAJE: $DATE" | tee -a "$LOG_FILE"
echo "------------------------------------------------------------" | tee -a "$LOG_FILE"

# Verificar si rclone está configurado
if ! rclone listremotes | grep -q "^$REMOTE_NAME:$"; then
    echo "[!] Error: El remoto '$REMOTE_NAME' no está configurado en rclone." | tee -a "$LOG_FILE"
    echo "[*] Ejecuta 'rclone config' para configurarlo."
    exit 1
fi

for FOLDER in "${FOLDERS[@]}"; do
    if [ -d "$FOLDER" ]; then
        echo "[>] Sincronizando: $FOLDER..." | tee -a "$LOG_FILE"
        
        # rclone sync: Hace que el destino sea un espejo exacto del origen
        # --drive-chunk-size: Optimizado para subidas grandes (modelos)
        rclone sync "$FOLDER" "$REMOTE_NAME:$REMOTE_PATH/$FOLDER" \
            --progress \
            --drive-chunk-size 64M \
            --log-file="$LOG_FILE" \
            --log-level INFO

        if [ $? -eq 0 ]; then
            echo "[✔] $FOLDER respaldado con éxito." | tee -a "$LOG_FILE"
        else
            echo "[X] Error al respaldar $FOLDER. Revisa los logs." | tee -a "$LOG_FILE"
        fi
    else
        echo "[!] Advertencia: La carpeta '$FOLDER' no existe. Saltando..." | tee -a "$LOG_FILE"
    fi
done

echo "------------------------------------------------------------" | tee -a "$LOG_FILE"
echo "[*] Respaldo finalizado a las $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"
echo "------------------------------------------------------------" | tee -a "$LOG_FILE"
