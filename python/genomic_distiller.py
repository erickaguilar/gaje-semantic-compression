import numpy as np
import gguf
import time
from genomize_llm import dequantize_q8_0, GenomicLLM
from transformers import AutoTokenizer


class GenomicDistiller:
    """
    Motor de Destilación para el Protocolo GAJE (Fase 10).
    Utiliza un modelo Maestro (F32) para refinar los centroides del Estudiante (2-bit).
    """

    def __init__(self, model_path, num_blocks=1):
        self.model_path = model_path
        self.num_blocks = num_blocks
        self.reader = gguf.GGUFReader(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")

        print("🧬 Distiller: Cargando modelo Maestro (F32)...")
        self.teacher = GenomicLLM(model_path, num_blocks=num_blocks, mode="f32")

    def collect_activations(self, prompt):
        """
        Recolecta activaciones reales del Maestro F32.
        """
        print(f"[*] Recolectando activaciones para: '{prompt}'")
        input_ids = self.tokenizer.encode(prompt)

        layer_stats = {}  # name -> vector sum

        # Procesar cada token
        for tid in input_ids:
            x = self.teacher.embedding_matrix[tid].tolist()
            for i, block in enumerate(self.teacher.blocks):
                prefix = f"blk.{i}."

                # 1. Stats para ffn_up y Attention (Input: 896)
                attn_out = block.attn.forward(x, 0)
                x_mid = block.rms_norm(np.array(x) + np.array(attn_out))

                # Usamos x_mid como proxy de entrada para la mayoría de capas
                if f"{prefix}input" not in layer_stats:
                    layer_stats[f"{prefix}input"] = np.zeros_like(x_mid)
                layer_stats[f"{prefix}input"] += np.abs(x_mid)

                # 2. Stats para ffn_down (Input: 4864)
                ffn_up_out = np.array(block.layers["ffn_up"].forward(x_mid))
                ffn_up_out = np.maximum(0, ffn_up_out)

                if f"{prefix}ffn_down" not in layer_stats:
                    layer_stats[f"{prefix}ffn_down"] = np.zeros_like(ffn_up_out)
                layer_stats[f"{prefix}ffn_down"] += np.abs(ffn_up_out)

                x = block.forward(x, 0)

        # Promediar
        for name in layer_stats:
            layer_stats[name] /= len(input_ids)
        return layer_stats

    def calibrate_layer_with_activations(
        self, teacher_weights_f32, student_thresholds, activation_stats, iterations=5
    ):
        """
        Optimización iterativa de centroides ponderada por importancia.
        """
        out_features, in_features = teacher_weights_f32.shape
        s = np.abs(activation_stats)
        s = s / (np.max(s) + 1e-12)

        final_centroids = []
        for i in range(out_features):
            w = teacher_weights_f32[i]
            t = student_thresholds[i]

            current_c = [0.0, 0.0, 0.0, 0.0]
            for _ in range(iterations):
                m0 = w < t[0]
                m1 = (w >= t[0]) & (w < t[1])
                m2 = (w >= t[1]) & (w < t[2])
                m3 = w >= t[2]

                def weighted_mean(vals, weights):
                    if not np.any(vals):
                        return 0.0
                    return np.sum(vals * weights) / (np.sum(weights) + 1e-12)

                current_c[0] = weighted_mean(w[m0], s[m0]) if np.any(m0) else t[0] - 0.1
                current_c[1] = (
                    weighted_mean(w[m1], s[m1]) if np.any(m1) else (t[0] + t[1]) / 2
                )
                current_c[2] = (
                    weighted_mean(w[m2], s[m2]) if np.any(m2) else (t[1] + t[2]) / 2
                )
                current_c[3] = weighted_mean(w[m3], s[m3]) if np.any(m3) else t[2] + 0.1

            final_centroids.extend(current_c)
        return final_centroids

    def run_distillation_pipeline(self, prompts, output_dir="gaje_v2_premium"):
        print("🚀 Iniciando Pipeline de Destilación Integral (Fase 10).")

        # 1. Estadísticas
        agg_stats = {}
        for p in prompts:
            stats = self.collect_activations(p)
            for name, val in stats.items():
                if name not in agg_stats:
                    agg_stats[name] = np.zeros_like(val)
                agg_stats[name] += val
        for name in agg_stats:
            agg_stats[name] /= len(prompts)

        # 2. Estudiante
        student = GenomicLLM(
            self.model_path, num_blocks=self.num_blocks, mode="genomic"
        )

        # 3. Destilación MHA + FFN
        start_time = time.time()
        for i in range(self.num_blocks):
            print(f"\n[*] Destilando Bloque {i} (MHA + FFN)...")
            block = student.blocks[i]
            prefix = f"blk.{i}."

            # A. Calibrar Atención (Q, K, V)
            attn_centroids = []
            for name in ["attn_q", "attn_k", "attn_v"]:
                tensor = next(
                    t
                    for t in self.reader.tensors
                    if t.name == prefix + name + ".weight"
                )
                w_m = dequantize_q8_0(tensor.data, tensor.shape[1], tensor.shape[0])
                t_list = []
                for row in w_m:
                    std = np.std(row)
                    mean = np.mean(row)
                    t_list.append([mean - 0.9816 * std, mean, mean + 0.9816 * std])

                # Importancia del input al bloque (896)
                stats = agg_stats.get(prefix + "input", np.ones(w_m.shape[1]))
                attn_centroids.extend(
                    self.calibrate_layer_with_activations(w_m, t_list, stats)
                )

            block.attn.kernel.centroids = attn_centroids

            # B. Calibrar FFN
            for name in ["ffn_up", "ffn_down"]:
                layer_full_name = prefix + name
                tensor = next(
                    t
                    for t in self.reader.tensors
                    if t.name == layer_full_name + ".weight"
                )
                w_m = dequantize_q8_0(tensor.data, tensor.shape[1], tensor.shape[0])
                t_list = []
                for row in w_m:
                    std = np.std(row)
                    mean = np.mean(row)
                    t_list.append([mean - 0.9816 * std, mean, mean + 0.9816 * std])

                stats = agg_stats.get(layer_full_name, agg_stats.get(prefix + "input"))
                block.layers[
                    name
                ].engine.centroids = self.calibrate_layer_with_activations(
                    w_m, t_list, stats
                )

        print(f"\n✅ Destilación finalizada en {time.time() - start_time:.2f}s")
        student.save_genomic_model(output_dir)
        print(f"🌟 Modelo Premium (MHA+FFN) guardado en: {output_dir}")


if __name__ == "__main__":
    model_path = "data/models/qwen2-0_5b-instruct-q8_0.gguf"
    distiller = GenomicDistiller(model_path, num_blocks=2)
    calibration_prompts = [
        "El protocolo GAJE es un sistema de compresión semántica",
        "La inteligencia artificial en dispositivos móviles requiere eficiencia extrema",
        "Rust y Python trabajando juntos permiten IA de alto rendimiento",
        "La compresión genémica de 2 bits preserva la intención del modelo original",
    ]
    distiller.run_distillation_pipeline(
        calibration_prompts, output_dir="gaje_v2_premium"
    )
