import os
import sys

# Asegurar que usamos el código local de 'python/'
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python"))
)

from gaje.nn.stabilized import GenomicLLM
from gaje.nn.configs import get_config


def main():
    print("🧬 GAJE PROTOCOL: BORN-GENOMIC MODEL INITIALIZATION")
    print("=" * 60)

    # 1. Obtener una configuración para el modelo nativo
    config = get_config("gaje_native")

    # 2. Inicializar el modelo desde el "nacimiento" (sin pesos pre-entrenados)
    # Definimos una arquitectura pequeña para el demo: 768 dim, 4 bloques
    model = GenomicLLM(num_blocks=4, config=config)

    print("\n[*] Modelo inicializado con éxito.")
    print(f"[*] Configuración: {model.config.name}")
    print(f"[*] Dimensiones: {model.n_embd}")
    print(f"[*] Bloques: {model.n_blocks}")

    # 3. Probar inferencia inicial (será aleatoria/ruido)
    prompt = "In the beginning was the code"
    print(f"\n[*] Prompt: '{prompt}'")
    print("[*] Generando (esperado: ruido inicial)...")

    for token in model.generate(prompt, max_new_tokens=10):
        print(token, end="", flush=True)
    print("\n")

    # 4. Explicación del siguiente paso: Entrenamiento
    print("=" * 60)
    print("🚀 SIGUIENTES PASOS:")
    print("1. El modelo ha nacido con un alfabeto genómico de 2 bits aleatorio.")
    print("2. Puedes usar model.refine_centroids() o model.refine_with_grads()")
    print("   para entrenar los centroides genómicos basándote en un dataset real.")
    print("3. Esto permite un control TOTAL sobre la entrada de pesos y la")
    print("   evolución del modelo desde cero bajo el protocolo GAJE.")
    print("=" * 60)


if __name__ == "__main__":
    main()
