import os
import sys
import json
import time
import argparse
import numpy as np

# Asegurar uso de código local
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python")))

import gaje.core._impl as engine
from gaje.nn.stabilized import GenomicLLM

def run_smg1_demo(model_path, prompt, max_tokens=20, mode="event"):
    print(f"🧬 PROTOCOLO SMG-1: Modo {mode.upper()}")
    print("-" * 60)
    
    if not os.path.exists(model_path):
        print(f"❌ Error: El modelo no existe en {model_path}")
        return

    # 1. Cargar metadatos y reconstruir capas
    reader = engine.GajeDatabaseReader(model_path)
    config = json.loads(reader.read_metadata("config"))
    centroides = config["centroides"]
    thresholds = config.get("thresholds", [0.4, 0.4, 0.4])
    
    layers = []
    for i, layer_cfg in enumerate(config["layers"]):
        l = engine.GajeNeuromorphicLayer(layer_cfg["out"], layer_cfg["in"], thresholds[i], 0.8)
        packed = reader.read_tensor(f"layer.{i}.packed_weights")
        l.load_packed_weights(packed)
        if mode == "event": l.k_wta = 1
        layers.append(l)
            
    # 2. Tokenizador
    teacher_path = "models/gguf/smollm2-135m-f16.gguf"
    if not os.path.exists(teacher_path):
        teacher_path = "models/gguf/qwen2-0_5b-q8_0.gguf"
    
    tokenizer = GenomicLLM(teacher_path).tokenizer
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if hasattr(tokens, "ids"): tokens = tokens.ids
    
    current_tokens = list(tokens)
    print(f"💬 Prompt: {prompt}")
    print("🤖 Generando: ", end="", flush=True)
    
    if mode == "event":
        scheduler = engine.NeuromorphicScheduler(centroides, 2)
        for _ in range(max_tokens):
            input_id = current_tokens[-1]
            scheduler.inject_spike(0, input_id, 0, 0, 1.0)
            next_id = None
            for tick in range(10):
                output_spikes = scheduler.step(layers)
                final_spikes = [s for s in output_spikes if s.target_layer_id >= len(layers)]
                if final_spikes:
                    next_id = final_spikes[0].source_neuron_id
                    break
            if next_id is None: next_id = (input_id + 1) % config["vocab_size"]
            print(tokenizer.decode([next_id]), end="", flush=True)
            current_tokens.append(next_id)
            for l in layers: l.apply_homeostasis(0.0)
    else:
        for _ in range(max_tokens):
            input_id = current_tokens[-1]
            layers[0].integrate_batch(input_id, centroides, 1.0)
            s0 = layers[0].check_spikes()
            if s0:
                for idx, intensity, _ in s0: layers[1].integrate_batch(idx, centroides, intensity)
                s1 = layers[1].check_spikes()
                if s1:
                    for idx, intensity, _ in s1: layers[2].integrate_batch(idx, centroides, intensity)
                    s2 = layers[2].check_spikes()
                    next_id = sorted(s2, key=lambda x: x[1], reverse=True)[0][0] if s2 else (input_id + 1) % config["vocab_size"]
                else: next_id = (input_id + 1) % config["vocab_size"]
            else: next_id = (input_id + 1) % config["vocab_size"]
            print(tokenizer.decode([next_id]), end="", flush=True)
            current_tokens.append(next_id)
            for l in layers: l.apply_homeostasis(0.0)
        
    print("\n\n✅ Protocolo SMG-1 Finalizado.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/checkpoints/smg1_overnight_gold_evolved.gaje")
    parser.add_argument("--prompt", type=str, default="ROMEO: ")
    parser.add_argument("--mode", choices=["event", "layer"], default="event")
    parser.add_argument("--tokens", type=int, default=30)
    args = parser.parse_args()
    run_smg1_demo(args.model, args.prompt, args.tokens, args.mode)
