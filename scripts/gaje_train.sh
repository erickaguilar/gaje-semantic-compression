#!/bin/bash

# 🧬 GAJE-Flow: Launcher de Entrenamiento Estabilizado
# Este script automatiza los ajustes de hilos y energía para evitar cierres de terminal.

# 1. Configuración de Parámetros
THREADS=2
ARCH="smollm"
BLOCKS=4
EPOCHS=35
GEN=20
DATASET="data/datasets/dataset_entrenamiento.txt"
MODEL_NAME="born_genomic_full"

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
echo -e "  [+] Epocas (Grad):     \033[1;37m$EPOCHS\033[0m"
echo -e "  [+] Generaciones (EVO):\033[1;37m$GEN\033[0m"
echo -e "  [+] Dataset:           \033[1;37m$DATASET\033[0m"
echo -e "\033[1;33m------------------------------------------------------------\033[0m"

# 3. Aplicar Ajustes
echo -e "[*] Solicitando Wake Lock..."
termux-wake-lock

echo -e "[*] Configurando variables de entorno..."
export RAYON_NUM_THREADS=$THREADS

# 4. Ejecutar Entrenamiento Híbrido
echo -e "[*] Iniciando protocolo híbrido (Gradientes + Evolución)...\n"
python scripts/train_born_genomic.py \
    --arch $ARCH \
    --blocks $BLOCKS \
    --epochs $EPOCHS \
    --evolve \
    --gen $GEN \
    --dataset $DATASET

# 5. Finalización
echo -e "\n\033[1;32m[✔] Entrenamiento completado con éxito.\033[0m"
echo -e "[*] Liberando Wake Lock..."
termux-wake-unlock
