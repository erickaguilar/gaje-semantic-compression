import os
import sys
import json
import numpy as np

# Asegurar uso de código local
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

import gaje.core._impl as engine
from gaje.nn.stabilized import GenomicLLM

def run_inference(model_path, prompt, max_tokens=20):
    print(f"🧬 Cargando modelo SMG-1 EVOLUCIONADO desde: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Error: El modelo no existe en {model_path}")
        return

    # 1. Cargar metadatos
    reader = engine.GajeDatabaseReader(model_path)
    config = json.loads(reader.read_metadata("config"))
    version = config.get("version", "unknown")
    centroides = config["centroides"]
    thresholds = config.get("thresholds", [0.4, 0.4, 0.4]) # Usar evolucionados o default
    
    print(f"[*] Versión: {version} | Centroides: {[round(c, 3) for c in centroides]}")
    print(f"[*] Umbrales: {[round(t, 3) for t in thresholds]}")
    
    # Reconstruir capas con umbrales y pesos evolucionados
    layers = []
    for i, layer_cfg in enumerate(config["layers"]):
        l = engine.GajeNeuromorphicLayer(layer_cfg["out"], layer_cfg["in"], thresholds[i], 0.8)
        packed = reader.read_tensor(f"layer.{i}.packed_weights")
        l.load_packed_weights(packed)
        layers.append(l)
            
    # 2. Tokenizador
    teacher_path = "models/gguf/smollm2-135m-f16.gguf"
    teacher = GenomicLLM(teacher_path)
    tokenizer = teacher.tokenizer

    print(f"\n💬 Prompt: {prompt}")
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if hasattr(tokens, "ids"): tokens = tokens.ids
    
    current_tokens = list(tokens)
    print("🤖 Generando: ", end="", flush=True)
    
    for _ in range(max_tokens):
        input_id = current_tokens[-1]
        
        # Simulación de Inferencia Nativa (3 Pasos de Spikes)
        layers[0].integrate_batch(input_id, centroides, 1.0)
        s0 = layers[0].check_spikes()
        
        if s0:
            for idx, intensity, _ in s0:
                layers[1].integrate_batch(idx, centroides, intensity)
            s1 = layers[1].check_spikes()
            
            if s1:
                for idx, intensity, _ in s1:
                    layers[2].integrate_batch(idx, centroides, intensity)
                s2 = layers[2].check_spikes()
                
                if s2:
                    # Elegir el spike con mayor intensidad (WTA simple)
                    next_id = sorted(s2, key=lambda x: x[1], reverse=True)[0][0]
                else:
                    next_id = (input_id + 1) % config["vocab_size"] # Fallback
            else:
                next_id = (input_id + 1) % config["vocab_size"]
        else:
            next_id = (input_id + 1) % config["vocab_size"]
        
        word = tokenizer.decode([next_id])
        print(word, end="", flush=True)
        current_tokens.append(next_id)
        
        # Reset homeostático para el siguiente token
        for l in layers:
            l.apply_homeostasis(0.0)
        
    print("\n\n✅ Inferencia Evolucionada Finalizada.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/checkpoints/smg1_overnight_gold_evolved.gaje")
    parser.add_argument("--prompt", type=str, default="ROMEO: ")
    parser.add_argument("--tokens", type=int, default=30)
    args = parser.parse_args()

    run_inference(
        args.model,
        args.prompt,
        max_tokens=args.tokens
    )
