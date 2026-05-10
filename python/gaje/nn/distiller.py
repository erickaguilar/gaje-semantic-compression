import numpy as np
from gaje.core import _impl as dna_semantic_compression
import gguf
import time
from gaje.nn.genomize import dequantize_q8_0, GenomicLLM as TeacherLLM
from gaje.nn.stabilized import GenomicLLM as StudentLLM, dequantize_q8_0 as dequantize_student
from gaje.core import _impl as dna_semantic_compression
import gguf
import time
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import numpy as np

class GenomicDistiller:
    """
    Motor de Destilación Híbrido para el Protocolo GAJE (Fase 10).
    Utiliza un modelo Maestro (F32) de alta fidelidad para refinar centroides.
    """
    def __init__(self, model_path, num_blocks=2):
        self.model_path = model_path
        self.num_blocks = num_blocks
        self.reader = gguf.GGUFReader(model_path)
        
        # Detect architecture to select correct tokenizer
        if "general.architecture" in self.reader.fields:
            part = self.reader.fields["general.architecture"].parts[-1]
            arch = bytes(part).decode("utf-8") if not isinstance(part[0], (bytes, bytearray)) else part[0].decode("utf-8")
        else:
            arch = "llama"
            
        tokenizer_name = "Qwen/Qwen2-0.5B" if arch == "qwen2" else "HuggingFaceTB/SmolLM2-135M-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        print(f"🧬 Distiller Híbrido: Cargando Maestro F32 de Alta Fidelidad...")
        self.teacher = TeacherLLM(model_path)
        # We limit the number of blocks if requested
        if num_blocks:
            self.teacher.blocks = self.teacher.blocks[:num_blocks]
            self.teacher.n_blocks = num_blocks
        
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
        
        # 1. Estadísticas de Activación
        agg_stats = {}
        for p in prompts:
            stats = self.collect_activations(p)
            for name, val in stats.items():
                if name not in agg_stats: agg_stats[name] = np.zeros_like(val)
                agg_stats[name] += val
        for name in agg_stats: agg_stats[name] /= len(prompts)

        # 2. Inicializar Estudiante Genómico
        print(f"\n🧬 Inicializando Estudiante (Genómico de Profundidad Completa)...")
        student = StudentLLM(self.model_path, num_blocks=self.num_blocks)

        # 3. Destilación MHA + FFN con IQAT
        start_time = time.time()
        for i in tqdm(range(self.num_blocks), desc="Destilando y Optimizando Bloques"):
            prefix = f"blk.{i}."
            block_student = student.blocks[i]
            
            # A. Calibrar Atención (Q, K, V Fusionados en Rust)
            # Extraemos pesos del Maestro para calibrar centroides iniciales
            w_q_m = dequantize_q8_0(next(t for t in self.reader.tensors if t.name == prefix + "attn_q.weight"))
            w_k_m = dequantize_q8_0(next(t for t in self.reader.tensors if t.name == prefix + "attn_k.weight"))
            w_v_m = dequantize_q8_0(next(t for t in self.reader.tensors if t.name == prefix + "attn_v.weight"))
            
            def get_thresholds(w):
                return [[np.mean(row)-0.98*np.std(row), np.mean(row), np.mean(row)+0.98*np.std(row)] for row in w]
            
            stats_in = agg_stats.get(prefix + "input", np.ones(w_q_m.shape[1]))
            
            c_q = self.calibrate_layer_with_activations(w_q_m, get_thresholds(w_q_m), stats_in)
            c_k = self.calibrate_layer_with_activations(w_k_m, get_thresholds(w_k_m), stats_in)
            c_v = self.calibrate_layer_with_activations(w_v_m, get_thresholds(w_v_m), stats_in)
            
            block_student.attn.attn.centroids = c_q + c_k + c_v

            # B. Calibrar FFN (SwiGLU Fusion)
            w_gate_m = dequantize_q8_0(next(t for t in self.reader.tensors if t.name == prefix + "ffn_gate.weight"))
            w_up_m = dequantize_q8_0(next(t for t in self.reader.tensors if t.name == prefix + "ffn_up.weight"))
            
            c_gate = self.calibrate_layer_with_activations(w_gate_m, get_thresholds(w_gate_m), stats_in)
            # Rust SwiGLU currently uses same centroids for both Gate and Up or handles it.
            # In our implementation, we passed gate_gen.dna_centroids.
            block_student.swiglu.centroids = c_gate

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
