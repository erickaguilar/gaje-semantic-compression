import numpy as np
import dna_semantic_compression
import gguf
import time
from python.genomize_llm import dequantize_q8_0, GenomicLLM
from transformers import AutoTokenizer
from tqdm import tqdm
import os

class GenomicDistiller:
    """
    Motor de Destilación Híbrido para el Protocolo GAJE (Fase 10).
    Utiliza un modelo Maestro (F32) de alta fidelidad para refinar centroides.
    """
    def __init__(self, model_path, num_blocks=2):
        self.model_path = model_path
        self.num_blocks = num_blocks
        self.reader = gguf.GGUFReader(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
        
        print(f"🧬 Distiller Híbrido: Cargando Maestro F32 de Alta Fidelidad...")
        # Usamos GenomicLLM en modo f32 pero con la lógica de full_genomic_pipeline
        # Para simplificar la implementación del maestro perfecto, 
        # nos aseguramos de que el forward pass incluya SiLU y RoPE.
        self.teacher = GenomicLLM(model_path, num_blocks=num_blocks, mode='f32')
        
    def collect_activations(self, prompt):
        """
        Recolecta activaciones reales del Maestro F32 con lógica de arquitectura completa.
        """
        print(f"[*] Recolectando activaciones (Maestro F32) para: '{prompt}'")
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        
        layer_stats = {} # name -> vector sum
        
        # Simulamos el forward pass del maestro recolectando estadísticas de magnitud
        for i, tid in enumerate(input_ids):
            x = self.teacher.embedding_matrix[tid].tolist()
            for b_idx, block in enumerate(self.teacher.blocks):
                prefix = f"blk.{b_idx}."
                
                # 1. Stats para Input de Atención y FFN
                x_arr = np.array(x)
                if f"{prefix}input" not in layer_stats: 
                    layer_stats[f"{prefix}input"] = np.zeros_like(x_arr)
                layer_stats[f"{prefix}input"] += np.abs(x_arr)
                
                # 2. Stats específicas para FFN (después de la proyección Up/Gate)
                # En Qwen2: SiLU(gate) * up
                # El gate suele ser el que define la "importancia" semántica
                x_mid = block.rms_norm(x_arr, 'attn_norm' if hasattr(block, 'layers') else None)
                
                # Forward de bloques (incluye RoPE y SiLU internamente en modo f32)
                x = block.forward(x, pos=i)
                    
        for name in layer_stats: layer_stats[name] /= len(input_ids)
        return layer_stats

    def calibrate_layer_with_activations(self, teacher_weights_f32, student_thresholds, activation_stats, iterations=5):
        """
        Optimización iterativa de centroides ponderada por importancia de la entrada.
        activation_stats debe tener la forma (in_features,)
        """
        out_features, in_features = teacher_weights_f32.shape
        # Asegurar que activation_stats coincida con in_features
        if len(activation_stats) != in_features:
            # Fallback a uniforme si hay desalineación (ej. capas de proyección)
            s = np.ones(in_features)
        else:
            s = np.abs(activation_stats)
            
        s = s / (np.max(s) + 1e-12)
        
        final_centroids = []
        for i in range(out_features):
            w = teacher_weights_f32[i] # (in_features,)
            t = student_thresholds[i]
            
            current_c = [0.0, 0.0, 0.0, 0.0]
            # Máscaras basadas en pesos
            m0 = w < t[0]; m1 = (w >= t[0]) & (w < t[1]); m2 = (w >= t[1]) & (w < t[2]); m3 = w >= t[2]
            
            def weighted_mean(vals, weights):
                if not np.any(vals): return 0.0
                return np.sum(vals * weights) / (np.sum(weights) + 1e-12)

            current_c[0] = weighted_mean(w[m0], s[m0]) if np.any(m0) else t[0] - 0.1
            current_c[1] = weighted_mean(w[m1], s[m1]) if np.any(m1) else (t[0] + t[1]) / 2
            current_c[2] = weighted_mean(w[m2], s[m2]) if np.any(m2) else (t[1] + t[2]) / 2
            current_c[3] = weighted_mean(w[m3], s[m3]) if np.any(m3) else t[2] + 0.1
                
            final_centroids.extend(current_c)
        return final_centroids

    def calculate_kl_divergence(self, p_logits, q_logits):
        """
        Calcula la divergencia KL entre la distribución del Maestro (p) y el Estudiante (q).
        """
        p = np.exp(p_logits - np.max(p_logits))
        p /= p.sum()
        q = np.exp(q_logits - np.max(q_logits))
        q /= q.sum()
        return np.sum(p * np.log((p + 1e-12) / (q + 1e-12)))

    def distill_logits(self, prompts):
        """
        Refinamiento de centroides basado en el error de predicción (Logit matching).
        """
        print(f"\n🚀 Iniciando Destilación de Logits (Logit Distillation)...")
        
        # 1. Cargar modelos
        student = GenomicLLM("gaje_qwen2_full_v1", load_genomic=True)
        
        for prompt in prompts:
            print(f"[*] Analizando discrepancia para: '{prompt}'")
            # Logits Maestro
            token_ids = self.tokenizer.encode(prompt)
            x_m = self.teacher.embedding_matrix[token_ids[-1]].tolist()
            for b in self.teacher.blocks: x_m = b.forward(x_m, 0)
            x_m = self.teacher.rms_norm(x_m, self.teacher.output_norm_weight)
            logits_m = np.dot(self.teacher.embedding_matrix, x_m)
            
            # Logits Estudiante
            x_s = student.embedding_matrix[token_ids[-1]].tolist()
            for b in student.blocks: x_s = b.forward(x_s, 0)
            x_s = student.rms_norm(x_s, student.output_norm_weight)
            logits_s = np.dot(student.embedding_matrix, x_s)
            
            kl = self.calculate_kl_divergence(logits_m, logits_s)
            print(f"    [!] Divergencia KL Inicial: {kl:.4f}")
            
            # En una versión avanzada, aquí ajustaríamos los centroides 
            # de las capas con mayor 'Activation Drift' para bajar el KL.
            # Por ahora, registramos la métrica para el roadmap.

    def run_distillation_pipeline(self, prompts, output_dir="gaje_qwen2_full_v1"):
        print(f"🚀 Iniciando Pipeline de Destilación Masiva (24 Bloques).")
        
        # 1. Estadísticas
        agg_stats = {}
        for p in prompts:
            stats = self.collect_activations(p)
            for name, val in stats.items():
                if name not in agg_stats: agg_stats[name] = np.zeros_like(val)
                agg_stats[name] += val
        for name in agg_stats: agg_stats[name] /= len(prompts)

        # 2. Estudiante
        print(f"\n🧬 Inicializando Estudiante (Genómico de Profundidad Completa)...")
        student = GenomicLLM(self.model_path, num_blocks=self.num_blocks, mode='genomic')

        # 3. Destilación MHA + FFN
        start_time = time.time()
        for i in tqdm(range(self.num_blocks), desc="Destilando Bloques"):
            block = student.blocks[i]
            prefix = f"blk.{i}."
            
            # A. Calibrar Atención
            attn_centroids = []
            for name in ["attn_q", "attn_k", "attn_v"]:
                tensor = next(t for t in self.reader.tensors if t.name == prefix + name + ".weight")
                w_m = dequantize_q8_0(tensor.data, tensor.shape[1], tensor.shape[0])
                t_list = [[np.mean(row)-0.98*np.std(row), np.mean(row), np.mean(row)+0.98*np.std(row)] for row in w_m]
                stats = agg_stats.get(prefix + "input", np.ones(w_m.shape[1]))
                attn_centroids.extend(self.calibrate_layer_with_activations(w_m, t_list, stats))
            block.attn.kernel.centroids = attn_centroids

            # B. Calibrar FFN
            for name in ["ffn_up", "ffn_down"]:
                layer_full_name = prefix + name
                tensor = next(t for t in self.reader.tensors if t.name == layer_full_name + ".weight")
                w_m = dequantize_q8_0(tensor.data, tensor.shape[1], tensor.shape[0])
                t_list = [[np.mean(row)-0.98*np.std(row), np.mean(row), np.mean(row)+0.98*np.std(row)] for row in w_m]
                stats = agg_stats.get(layer_full_name, agg_stats.get(prefix + "input"))
                block.layers[name].engine.centroids = self.calibrate_layer_with_activations(w_m, t_list, stats)

        print(f"\n✅ Destilación finalizada en {time.time() - start_time:.2f}s")
        student.save_genomic_model(output_dir)
        print(f"🌟 MODELO COMPLETO GUARDADO EN: {output_dir}")

if __name__ == "__main__":
    model_path = "/data/data/com.termux/files/home/models/qwen2-0_5b-q8_0.gguf"
    # DISTILACIÓN QUIRÚRGICA: Solo 2 bloques para validar el enfoque híbrido
    num_test_blocks = 2
    distiller = GenomicDistiller(model_path, num_blocks=num_test_blocks)
    
    calibration_prompts = [
        "El protocolo GAJE es un sistema de compresión semántica de 2 bits.",
        "La inteligencia artificial en dispositivos móviles requiere eficiencia extrema.",
        "Rust y Python trabajando juntos permiten IA de alto rendimiento.",
        "La compresión genómica preserva la intención del modelo original."
    ]
    
    output_dir = "gaje_qwen2_hybrid_v1"
    distiller.run_distillation_pipeline(calibration_prompts, output_dir=output_dir)
    
    print(f"\n🚀 Destilación Híbrida Completada. Directorio: {output_dir}")
    print("[*] Siguiente paso: Ejecutar benchmarks/distilled_qwen_test.py apuntando a este modelo.")
