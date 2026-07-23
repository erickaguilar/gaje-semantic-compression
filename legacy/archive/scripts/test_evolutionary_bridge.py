import os
import sys
import time

# Ensure we use the local package first
sys.path.insert(0, os.path.abspath("python"))

from gaje.nn.stabilized import GenomicLLM
from gaje.nn.configs import get_config
from gaje.nn.trainer import GenomicTrainer


def main():
    print("🧬 GAJE PROTOCOL: EVOLUTIONARY LEARNING TEST")
    print("=" * 60)

    # 1. Configuración de Arquitectura (Ligera para test)
    config = get_config("smollm")

    # 2. Inicialización
    llm = GenomicLLM(model_path=None, num_blocks=2, config=config)

    # 3. Dataset de prueba
    dataset = [
        "El protocolo GAJE es nativo.",
        "El protocolo GAJE comprime semántica.",
        "GAJE utiliza ADN en lugar de pesos.",
    ]

    # 4. Iniciar Entrenador
    trainer = GenomicTrainer(llm)

    # 5. Fase 1: Entrenamiento por Gradientes (Poco, solo para estabilizar)
    print("\n[*] Fase 1: Estabilización por Gradientes (2 épocas)...")
    trainer.fit(dataset, epochs=2)

    # 6. Fase 2: Evolución Genética (Homeostasis)
    print("\n[*] Fase 2: Evolución Genética (20 generaciones)...")
    start_time = time.time()
    trainer.evolve(dataset, generations=20, mutation_scale=0.05)
    duration = time.time() - start_time

    print(f"\n[*] Evolución finalizada en {duration:.2f} segundos.")

    # 7. Generación final
    prompt = "El protocolo GAJE"
    print("\n[*] Resultado final:")
    print("🤖 GAJE: ", end="", flush=True)
    for token in llm.generate(prompt, max_new_tokens=10):
        print(token, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    main()
