#!/bin/bash

# 🧬 GAJE-Flow: Launcher de Destilación Genómica
# Este script comprime un modelo inteligente (SmolLM2) al formato GAJE de 2 bits.

# 1. Configuración de Parámetros
THREADS=2
SOURCE_MODEL="models/gguf/smollm2-135m-q8_0.gguf"
NAME="GajeSmol-v1"
EPOCHS=5 # Pocas épocas porque el modelo ya es inteligente

# 2. Interfaz Visual
clear
echo -e "\033[1;35m"
echo "      ::::::::      :::          ::: ::::::::::: :::::::::: "
echo "    :+:    :+:    :+: :+:        :+:     :+:     :+:        "
echo "   +:+           +:+   +:+       +:+     +:+     +:+         "
echo "  :#:          +#++:++#++:      +#+     +#+     +#++:++#     "
echo " +#+   #     +#+     +#+      +#+     +#+     +#+            "
echo "#+#    #    #+#     #+#  #+# #+#     #+#     #+#             "
echo "########    ###     ###   #####  ########### ##########      "
echo -e "\033[0m"
echo -e "\033[1;33m------------------------------------------------------------\033[0m"
echo -e "\033[1;34m  PROTOCOLO DE DESTILACIÓN (GGUF -> GAJE):\033[0m"
echo -e "  [+] Modelo Fuente:     \033[1;37m$SOURCE_MODEL\033[0m"
echo -e "  [+] Nombre Destino:    \033[1;37m$NAME\033[0m"
echo -e "  [+] Hilos (Rayon):     \033[1;37m$THREADS\033[0m"
echo -e "  [+] Refinamiento:      \033[1;37m$EPOCHS épocas\033[0m"
echo -e "\033[1;33m------------------------------------------------------------\033[0m"

# 3. Verificación de Fuente
if [ ! -L "$SOURCE_MODEL" ] && [ ! -f "$SOURCE_MODEL" ]; then
    echo -e "\033[1;31m[!] Error: No se encuentra el modelo fuente en $SOURCE_MODEL\033[0m"
    exit 1
fi

# 4. Aplicar Ajustes de Estabilidad
echo -e "[*] Solicitando Wake Lock..."
termux-wake-lock
export RAYON_NUM_THREADS=$THREADS

# 5. Ejecutar Destilación
echo -e "[*] Iniciando destilación genómica...\n"
python scripts/distill_smollm.py \
    --source "$SOURCE_MODEL" \
    --name "$NAME" \
    --epochs $EPOCHS

# 6. Finalización
echo -e "\n\033[1;32m[✔] Destilación completada con éxito.\033[0m"
echo -e "[*] Liberando Wake Lock..."
termux-wake-unlock
