#!/bin/bash

# 🧬 GAJE-Flow: Launcher de Entrenamiento Estabilizado
# Este script automatiza los ajustes de hilos y energía para evitar cierres de terminal.

# 1. Configuración de Parámetros
THREADS=2
ARCH="smollm"
BLOCKS=2
EPOCHS=10
MODEL_NAME="born_genomic_smollm"

# 2. Interfaz Visual
clear
echo -e "\033[1;36m"
echo "      ::::::::      :::          ::: ::::::::::: :::::::::: "
echo "    :+:    :+:    :+: :+:        :+:     :+:     :+:        "
echo "   +:+           +:+   +:+       +:+     +:+     +:+         "
echo "  :#:          +#++:++#++:      +#+     +#+     +#++:++#     "
echo " +#+   #     +#+     +#+      +#+     +#+     +#+            "
echo "#+#    #    #+#     #+#  #+# #+#     #+#     #+#             "
echo "########    ###     ###   #####  ########### ##########      "
echo -e "\033[0m"
echo -e "\033[1;33m------------------------------------------------------------\033[0m"
echo -e "\033[1;32m  PARAMETROS DE ESTABILIZACION:\033[0m"
echo -e "  [+] Hilos (Rayon):     \033[1;37m$THREADS\033[0m"
echo -e "  [+] Wake Lock:         \033[1;37mACTIVADO\033[0m"
echo -e "  [+] Arquitectura:      \033[1;37m$ARCH\033[0m"
echo -e "  [+] Bloques:           \033[1;37m$BLOCKS\033[0m"
echo -e "  [+] Epocas:            \033[1;37m$EPOCHS\033[0m"
echo -e "\033[1;33m------------------------------------------------------------\033[0m"

# 3. Aplicar Ajustes
echo -e "[*] Solicitando Wake Lock..."
termux-wake-lock

echo -e "[*] Configurando variables de entorno..."
export RAYON_NUM_THREADS=$THREADS

# 4. Ejecutar Entrenamiento
echo -e "[*] Iniciando protocolo de entrenamiento...\n"
python scripts/train_born_genomic.py \
    --arch $ARCH \
    --blocks $BLOCKS \
    --epochs $EPOCHS

# 5. Finalización
echo -e "\n\033[1;32m[✔] Entrenamiento completado con éxito.\033[0m"
echo -e "[*] Liberando Wake Lock..."
termux-wake-unlock
