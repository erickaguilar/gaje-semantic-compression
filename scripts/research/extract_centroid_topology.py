import os
import sys

# Use local mock scipy to bypass broken system installation
sys.path.insert(0, os.path.abspath(".mock_scipy"))

import torch
import numpy as np
import json
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Asegurar uso de código local
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python")))

# --- INFORMACIÓN DE DATASETS (Carpeta data/datasets/) ---
# El proceso de extracción de topología depende de la "imprimación" semántica. 
# Datasets disponibles para usar con el flag --data:
# 1. expert_rust.txt: Solo conceptos profundos de Rust. Ideal para validar traslado técnico.
# 2. coherence_es.txt: Lógica y gramática pura en español. Ideal para validar estabilidad lingüística.
# 2. dataset_es_ext.txt: Versión extendida para mayor cobertura semántica.
# 3. tiny_shakespeare.txt: Recomendado para extraer estructuras literarias y gramática compleja (Inglés).
# 4. hybrid_polyglot_dataset.txt: Útil para topologías bilingües y detección de cambio de código.
# 5. dataset_born_2000.txt: Datos sintéticos para validación de ruido genómico.
# ---------------------------------------------------------

class TopologyExtractor:
    def __init__(self, model_id):
        print(f"[*] Cargando Maestro para Extracción: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float32, 
            output_hidden_states=True
        )
        self.model.eval()
        
    def extract_cam(self, text_corpus, num_samples=100, seq_len=64):
        print(f"[*] Extrayendo Topología Relacional (Corpus: {len(text_corpus)} líneas)...")
        
        # Estructura de la CAM: [capa_i][centroide_a][centroide_b]
        # Simplificación: Usaremos 4 "Estados de Activación" (Clusters) para coincidir con 2-bits
        num_layers = len(self.model.model.layers)
        cam = {layer_idx: np.zeros((4, 4)) for layer_idx in range(num_layers - 1)}
        
        samples = text_corpus[:num_samples]
        
        for text in tqdm(samples, desc="Mapeando activaciones"):
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
            with torch.no_grad():
                outputs = self.model(**inputs)
                # hidden_states: [num_layers + 1][batch, seq, hidden]
                h_states = outputs.hidden_states
                
            # Analizar flujo entre capas
            for i in range(num_layers - 1):
                # Tomar la activación promedio del bloque de tokens para ver la tendencia de la capa
                act_i = h_states[i+1][0].numpy() # Capa i
                act_next = h_states[i+2][0].numpy() # Capa i+1
                
                # Cuantizar activaciones en 4 niveles (basado en cuantiles para distribución uniforme)
                def quantize_activations(act):
                    q = np.quantile(act, [0.25, 0.5, 0.75])
                    q_act = np.zeros_like(act, dtype=np.int8)
                    q_act[act > q[0]] = 1
                    q_act[act > q[1]] = 2
                    q_act[act > q[2]] = 3
                    return q_act

                q_i = quantize_activations(act_i)
                q_next = quantize_activations(act_next)
                
                # Registrar transiciones
                # Para cada par de dimensiones correlacionadas (simplificado a transiciones directas)
                for t in range(min(q_i.shape[0], q_next.shape[0])):
                    # Tomamos el estado predominante de la activación en ese paso de tiempo
                    state_a = int(np.median(q_i[t]))
                    state_b = int(np.median(q_next[t]))
                    cam[i][state_a, state_b] += 1

        # Normalizar para obtener probabilidades
        final_cam = {}
        for layer_idx, matrix in cam.items():
            row_sums = matrix.sum(axis=1)
            # Evitar división por cero
            norm_matrix = matrix / (row_sums[:, np.newaxis] + 1e-12)
            final_cam[str(layer_idx)] = norm_matrix.tolist()
            
        return final_cam

def run_step_1(model_id, data_path, output_path):
    extractor = TopologyExtractor(model_id)
    
    if not os.path.exists(data_path):
        print(f"❌ Error: Dataset no encontrado en {data_path}")
        return
        
    with open(data_path, "r", encoding="utf-8") as f:
        corpus = [line.strip() for line in f.readlines() if len(line.strip()) > 20]
        
    topology = extractor.extract_cam(corpus)
    
    # Guardar mapa topológico
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_source": model_id,
            "type": "CentroidAdjacencyMatrix",
            "states": 4,
            "topology": topology
        }, f, indent=2)
        
    print(f"\n✅ Mapa Topológico extraído exitosamente en: {output_path}")
    print(f"[*] Capas procesadas: {len(topology)}")
    print(f"[*] Próximo paso: Inyectar este mapa en el motor de Rust.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paso 1: Extracción de Topología Relacional")
    parser.add_argument("--master", type=str, default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--data", type=str, default="data/datasets/dataset_es.txt")
    parser.add_argument("--output", type=str, default="models/core/topology_map.json")
    args = parser.parse_args()

    run_step_1(args.master, args.data, args.output)
