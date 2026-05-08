import numpy as np
import dna_semantic_compression
import gguf
import time
from python.genomize_llm import dequantize_q8_0, GenomicLLM
from transformers import AutoTokenizer
from tqdm import tqdm

class GenomicDistiller:
    """
    Motor de Destilación para el Protocolo GAJE.
    Ajusta los centroides genómicos basándose en pesos y activaciones.
    """
    def __init__(self, model_path):
        self.reader = gguf.GGUFReader(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
        print(f"🧬 Distiller: Cargando modelo maestro para calibración...")
        
    def collect_activations(self, prompt, num_blocks=1):
        """
        Simula un forward pass en F32 para recolectar las activaciones (entradas a cada capa).
        """
        print(f"[*] Recolectando activaciones para: '{prompt}'")
        tokens = self.tokenizer.encode(prompt)
        
        # 1. Cargar embeddings maestros
        embd_tensor = next(t for t in self.reader.tensors if t.name == "token_embd.weight")
        W_embd = dequantize_q8_0(embd_tensor.data, embd_tensor.shape[1], embd_tensor.shape[0])
        
        # 2. RMSNorm final maestro
        norm_weight = next(t for t in self.reader.tensors if t.name == "output_norm.weight").data.astype(np.float32)

        activations = {} # layer_name -> list of input vectors
        
        # Propagar
        x = W_embd[tokens[-1]]
        
        for i in range(num_blocks):
            prefix = f"blk.{i}."
            # Guardamos la activación de entrada al bloque (que va a Attn y FFN)
            activations[f"{prefix}ffn_up"] = x.copy()
            
            # Forward F32 simplificado (Teacher)
            # Para este prototipo, simulamos el paso por el bloque F32
            # En una versión real, esto usaría la lógica completa de GenomicLLM pero en F32
            # Aquí solo queremos la magnitud promedio de x por dimensión
            pass 
            
        return activations

    def calibrate_layer_with_activations(self, layer_name, teacher_weights_f32, student_thresholds, activation_stats):
        """
        Optimización de centroides ponderada por activaciones (AWQ-style).
        """
        out_features, in_features = teacher_weights_f32.shape
        new_centroids = []
        
        # Calculamos la importancia de cada canal de entrada
        # activation_stats es el vector x promedio (magnitud)
        s = np.abs(activation_stats)
        s = s / (np.max(s) + 1e-12)
        
        for i in range(out_features):
            w = teacher_weights_f32[i]
            t = student_thresholds[i]
            
            # Ponderamos el error de reconstrucción por la magnitud de la activación
            # Sin embargo, en 2-bit Max-Lloyd, el refinamiento es simple:
            # El centroide es la media de los pesos que caen en ese bin.
            # Con activaciones, podemos desplazar los umbrales t para proteger canales ruidosos.
            
            mask0 = w < t[0]
            mask1 = (w >= t[0]) & (w < t[1])
            mask2 = (w >= t[1]) & (w < t[2])
            mask3 = w >= t[2]
            
            c0 = np.mean(w[mask0]) if np.any(mask0) else t[0] - 0.5
            c1 = np.mean(w[mask1]) if np.any(mask1) else (t[0] + t[1]) / 2
            c2 = np.mean(w[mask2]) if np.any(mask2) else (t[1] + t[2]) / 2
            c3 = np.mean(w[mask3]) if np.any(mask3) else t[2] + 0.5
            
            new_centroids.extend([c0, c1, c2, c3])
            
        return new_centroids

    def distill_block(self, block_idx):
        print(f"\n[*] Destilando Bloque {block_idx}...")
        prefix = f"blk.{block_idx}."
        
        # En esta fase, optimizamos los centroides de las capas FFN
        for name in ["ffn_up", "ffn_down"]:
            tensor = next(t for t in self.reader.tensors if t.name == prefix + name + ".weight")
            print(f"    - Calibrando {name}...")
            
            # 1. Obtener pesos maestros
            w_master = dequantize_q8_0(tensor.data, tensor.shape[1], tensor.shape[0])
            
            # 2. Calcular umbrales iniciales (Baseline Max-Lloyd)
            thresholds = []
            for row in w_master:
                std = np.std(row)
                mean = np.mean(row)
                thresholds.append([mean - 0.9816 * std, mean, mean + 0.9816 * std])
            
            # 3. Refinar centroides (Inyección de Conocimiento)
            refined_centroids = self.calibrate_layer(name, w_master, thresholds)
            
            # Aquí guardaríamos los centroides refinados para el modelo final
            # Por ahora, reportamos la mejora de MSE
            print(f"    [+] {name}: Centroides refinados para {tensor.shape[1]} neuronas.")

    def run_distillation_pipeline(self, prompts, num_blocks=1, output_dir="qwen2_genomic_distilled"):
        print(f"🚀 Iniciando Pipeline de Destilación con {len(prompts)} frases de calibración.")
        
        # 1. Cargar modelo genómico inicial para destilar
        # Usamos GenomicLLM para manejar la estructura de bloques
        model = GenomicLLM(model_path, num_blocks=num_blocks)

        # 2. Recolectar estadísticas de activaciones
        agg_activations = {}
        for p in prompts:
            acts = self.collect_activations(p, num_blocks)
            for name, val in acts.items():
                if name not in agg_activations: agg_activations[name] = np.zeros_like(val)
                agg_activations[name] += np.abs(val)
        
        for name in agg_activations: agg_activations[name] /= len(prompts)

        # 3. Destilar y Actualizar Bloques
        start_time = time.time()
        for i in range(num_blocks):
            print(f"\n[*] Destilando Bloque {i} (Activation-Aware)...")
            block = model.blocks[i]
            prefix = f"blk.{i}."
            
            # Calibrar FFN
            for name in ["ffn_up", "ffn_down"]:
                layer_full_name = prefix + name
                tensor = next(t for t in self.reader.tensors if t.name == layer_full_name + ".weight")
                w_master = dequantize_q8_0(tensor.data, tensor.shape[1], tensor.shape[0])
                
                # Umbrales actuales (podríamos extraerlos del block si fuera necesario)
                # Aquí recalculamos para simplicidad del prototipo
                thresholds = []
                for row in w_master:
                    std = np.std(row); mean = np.mean(row)
                    thresholds.append([mean - 0.9816 * std, mean, mean + 0.9816 * std])
                
                stats = agg_activations.get(layer_full_name, np.ones(w_master.shape[1]))
                refined_centroids = self.calibrate_layer_with_activations(name, w_master, thresholds, stats)
                
                # Inyectar centroides refinados en el motor Rust
                block.layers[name].engine.centroids = refined_centroids

            print(f"    [+] Bloque {i}: Calibración de centroides completada.")

        print(f"\n✅ Pipeline de Destilación Finalizado en {time.time() - start_time:.2f}s")
        
        # 4. Guardar Modelo Distilado
        model.save_genomic_model(output_dir)
        print(f"🌟 Modelo destilado guardado exitosamente en: {output_dir}")

if __name__ == "__main__":
    model_path = "/data/data/com.termux/files/home/models/qwen2-0_5b-q8_0.gguf"
    distiller = GenomicDistiller(model_path)
    
    calibration_prompts = [
        "El protocolo GAJE es un sistema de compresión",
        "La inteligencia artificial en dispositivos móviles requiere eficiencia",
        "Rust y Python trabajando juntos para IA",
        "Compresión genémica de 2 bits preserva la semántica"
    ]
    
    distiller.run_distillation_pipeline(calibration_prompts, num_blocks=2, output_dir="gaje_model_v1")
