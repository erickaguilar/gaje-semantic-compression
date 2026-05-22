import os
import sys
import json
import numpy as np

# Asegurar uso de código local
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

import gaje.core._impl as engine
from gaje.nn.stabilized import GenomicLLM

def run_inference(model_path, prompt, max_tokens=20):
    print(f"🧬 Cargando modelo SMG-1 desde: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Error: El modelo no existe en {model_path}")
        return

    # 1. Cargar metadatos y capas
    reader = engine.GajeDatabaseReader(model_path)
    config = json.loads(reader.read_metadata("config"))
    version = config.get("version", "unknown")
    vocab_size = config["vocab_size"]
    dim_latent = config["dim_latent"]
    dim_logic = config["dim_logic"]
    centroides = config["centroides"]
    
    print(f"[*] Modelo detectado: {config['type']} (v{version})")
    
    # Reconstruir capas
    layers = []
    for i in range(len(config["layers"])):
        layer_cfg = config["layers"][i]
        # Crear capa con dimensiones originales
        l = engine.GajeNeuromorphicLayer(layer_cfg["out"], layer_cfg["in"], 0.4, 0.8)
        # Cargar pesos (descomprimidos automáticamente por el Reader si usamos lz4)
        packed = reader.read_tensor(f"layer.{i}.packed_weights")
        # Nota: Idealmente el GajeNeuromorphicLayer debería tener un setter para packed_weights
        # Si no lo tiene, este script servirá de guía para añadirlo.
        # Por ahora, simularemos la lógica si el binding lo permite.
        try:
            # Intentar cargar pesos si el binding lo soporta (check internal)
            # l.packed_weights = packed 
            # Si no hay setter, necesitaremos una pequeña modificación en Rust o usar set_weight
            # Pero para esta prueba, vamos a intentar usar el motor de inferencia estándar
            # si el modelo fuera un GenomicLLM (Transformer). 
            # El SMG-1 es experimental, así que usaremos integración manual.
            pass
        except Exception as e:
            print(f"[*] Aviso: Binding de carga directa no disponible, usando modo simulación: {e}")
            
    # 2. Tokenizador (Usamos el del maestro para esta prueba)
    # En un sistema real, el SMG-1 tendría su propio tokenizador o ID de referencia.
    teacher_path = "models/gguf/smollm2-135m-f16.gguf"
    teacher = GenomicLLM(teacher_path)
    tokenizer = teacher.tokenizer

    print(f"\n💬 Prompt: {prompt}")
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if hasattr(tokens, "ids"): tokens = tokens.ids
    
    generated_text = prompt
    current_tokens = list(tokens)
    
    print("🤖 Generando: ", end="", flush=True)
    
    # Bucle de generación (Simplificado para SMG-1)
    for _ in range(max_tokens):
        # En el SMG-1 nativo, esto sería un forward de impulsos
        # Para la prueba, simularemos la activación de la última capa
        
        # 1. Input ID
        last_id = current_tokens[-1]
        
        # 2. Disparo de capas (Simulado vía refine_step o lógica similar si no hay forward directo)
        # Como el SMG-1 es un prototipo, el motor de inferencia spiking completo está en src/nn/spiking/
        # Usaremos el motor nativo si está disponible.
        
        # Para esta demo, seleccionamos el token con más probabilidad (simulando spikes)
        # En una implementación real, aquí llamaríamos a rust_llm.forward_spiking()
        
        # Simulamos una respuesta coherente basada en el entrenamiento de Shakespeare
        next_id = (last_id + 7) % vocab_size # Placeholder para ver el flujo
        
        word = tokenizer.decode([next_id])
        print(word, end="", flush=True)
        generated_text += word
        current_tokens.append(next_id)
        
    print("\n\n✅ Prueba de flujo finalizada.")

if __name__ == "__main__":
    run_inference(
        "models/checkpoints/smollm2_distilled_smg1.gaje",
        "ROMEO: ",
        max_tokens=15
    )
