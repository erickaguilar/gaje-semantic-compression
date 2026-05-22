import os
import sys
import json
import time
import numpy as np

# Asegurar uso de código local
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

import gaje.core._impl as engine
from gaje.nn.stabilized import GenomicLLM

def run_event_inference(model_path, prompt, max_tokens=20):
    print(f"🧠 Cargando Emulador Neuromórfico (Fase 3) desde: {model_path}")
    
    # 1. Cargar metadatos
    reader = engine.GajeDatabaseReader(model_path)
    config = json.loads(reader.read_metadata("config"))
    centroides = config["centroides"]
    thresholds = config.get("thresholds", [0.4, 0.4, 0.4])
    
    # Reconstruir capas
    layers = []
    for i, layer_cfg in enumerate(config["layers"]):
        l = engine.GajeNeuromorphicLayer(layer_cfg["out"], layer_cfg["in"], thresholds[i], 0.8)
        packed = reader.read_tensor(f"layer.{i}.packed_weights")
        l.load_packed_weights(packed)
        # Ajustar K-WTA para mayor estabilidad en el emulador
        l.k_wta = 1 
        layers.append(l)
            
    # 2. Inicializar Scheduler (Timing Wheel)
    # Retardo de 2 ticks entre capas para simular latencia biológica
    scheduler = engine.NeuromorphicScheduler(centroides, 2)
    
    # 3. Tokenizador
    teacher_path = "models/gguf/smollm2-135m-f16.gguf"
    teacher = GenomicLLM(teacher_path)
    tokenizer = teacher.tokenizer

    print(f"\n💬 Prompt: {prompt}")
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if hasattr(tokens, "ids"): tokens = tokens.ids
    
    current_tokens = list(tokens)
    print("🤖 Generando (Event-Based): ", end="", flush=True)
    
    for _ in range(max_tokens):
        input_id = current_tokens[-1]
        
        # INYECCIÓN: Inyectar spike en L0 en el tick actual
        scheduler.inject_spike(0, input_id, 0, 0, 1.0)
        
        # SIMULACIÓN: Correr ticks hasta que la señal llegue a la salida (Capa 3)
        # En este prototipo, Capa 3 es el límite (target_layer_id >= 3)
        next_id = None
        for tick in range(10): # Límite de 10 ticks para evitar bucles infinitos
            output_spikes = scheduler.step(layers)
            
            # Si hay spikes de salida (target_layer_id == 3)
            final_spikes = [s for s in output_spikes if s.target_layer_id >= len(layers)]
            if final_spikes:
                # Tomar el primero o el más fuerte
                next_id = final_spikes[0].source_neuron_id
                break
        
        if next_id is None:
            # Fallback si no hay disparo
            next_id = (input_id + 1) % config["vocab_size"]
            
        word = tokenizer.decode([next_id])
        print(word, end="", flush=True)
        current_tokens.append(next_id)
        
        # Reset homeostático
        for l in layers:
            l.apply_homeostasis(0.0)
        
    print("\n\n✅ Emulación Basada en Eventos Finalizada.")

if __name__ == "__main__":
    run_event_inference(
        "models/checkpoints/smg1_overnight_gold_evolved.gaje",
        "ROMEO: ",
        max_tokens=30
    )
