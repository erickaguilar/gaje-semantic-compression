import numpy as np
import gguf
import time
from gaje.nn.genomize import dequantize_q8_0, GenomicLLM as TeacherLLM
from gaje.nn.stabilized import GenomicLLM as StudentLLM
from transformers import AutoTokenizer
from tqdm import tqdm


class GenomicDistiller:
    """
    Motor de Destilación por Consejo (Council of Teachers) - Fase 11.4.
    Consolida conocimiento de múltiples maestros para estabilizar el estudiante genómico.
    """

    def __init__(self, model_paths, num_blocks=2):
        if isinstance(model_paths, str):
            model_paths = [model_paths]

        self.model_paths = model_paths
        self.num_blocks = num_blocks

        # Usamos el primer modelo para extraer metadatos y tokenizador
        self.reader = gguf.GGUFReader(model_paths[0])

        if "general.architecture" in self.reader.fields:
            part = self.reader.fields["general.architecture"].parts[-1]
            arch = (
                bytes(part).decode("utf-8")
                if not isinstance(part[0], (bytes, bytearray))
                else part[0].decode("utf-8")
            )
        else:
            arch = "llama"

        tokenizer_name = (
            "Qwen/Qwen2-0.5B"
            if arch == "qwen2"
            else "HuggingFaceTB/SmolLM2-135M-Instruct"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        print(f"🧬 Council of Teachers: Cargando {len(model_paths)} Maestros F32...")
        self.teachers = []
        for path in model_paths:
            t = TeacherLLM(path)
            if num_blocks:
                t.blocks = t.blocks[:num_blocks]
                t.n_blocks = num_blocks
            self.teachers.append(t)

    def collect_activations(self, prompt):
        """
        Recolecta activaciones promediadas de todo el Consejo de Maestros.
        """
        print(f"[*] Recolectando Consenso para: '{prompt}'")
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        council_stats = {}  # name -> vector sum

        for t_idx, teacher in enumerate(self.teachers):
            print(f"    [~] Consultando Maestro {t_idx+1}/{len(self.teachers)}...")
            for i, tid in enumerate(input_ids):
                # Usamos el embedding del maestro actual
                x = teacher.embedding_matrix[tid].tolist()
                for b_idx, block in enumerate(teacher.blocks):
                    prefix = f"blk.{b_idx}."

                    x_arr = np.array(x)
                    if f"{prefix}input" not in council_stats:
                        council_stats[f"{prefix}input"] = np.zeros_like(x_arr)

                    # Activación ponderada por el número de maestros
                    council_stats[f"{prefix}input"] += np.abs(x_arr) / len(
                        self.teachers
                    )

                    # Forward
                    x = block.forward(x, pos=i)

        for name in council_stats:
            council_stats[name] /= len(input_ids)
        return council_stats

    def run_distillation_pipeline(self, prompts, output_dir="gaje_council_model"):
        print(
            f"🚀 Iniciando Pipeline de Destilación por Consejo ({len(self.teachers)} Maestros)."
        )

        # 1. Consenso de Activaciones
        agg_stats = {}
        for p in prompts:
            stats = self.collect_activations(p)
            for name, val in stats.items():
                if name not in agg_stats:
                    agg_stats[name] = np.zeros_like(val)
                agg_stats[name] += val
        for name in agg_stats:
            agg_stats[name] /= len(prompts)

        # 2. Inicializar Estudiante (usando el primer modelo como base estructural)
        print("\n🧬 Inicializando Estudiante (Genómico de Consenso)...")
        student = StudentLLM(self.model_paths[0], num_blocks=self.num_blocks)

        # 3. Destilación Multi-Maestro
        start_time = time.time()
        for i in tqdm(range(self.num_blocks), desc="Destilando Bloques"):
            prefix = f"blk.{i}."
            block_student = student.blocks[i]

            # Ponderación de pesos de todos los maestros
            # (En esta fase, promediamos pesos F32 antes de convertirlos a ADN)
            w_q_agg = None
            for teacher_reader in [gguf.GGUFReader(p) for p in self.model_paths]:
                w = dequantize_q8_0(
                    next(
                        t
                        for t in teacher_reader.tensors
                        if t.name == prefix + "attn_q.weight"
                    )
                )
                if w_q_agg is None:
                    w_q_agg = np.zeros_like(w)
                w_q_agg += w / len(self.model_paths)

            # Calibración basada en activaciones del consejo
            stats_in = agg_stats.get(prefix + "input", np.ones(w_q_agg.shape[1]))

            def get_thresholds(w):
                return [
                    [
                        np.mean(row) - 0.98 * np.std(row),
                        np.mean(row),
                        np.mean(row) + 0.98 * np.std(row),
                    ]
                    for row in w
                ]

            block_student.attn.attn.centroids = self.calibrate_layer_with_activations(
                w_q_agg, get_thresholds(w_q_agg), stats_in
            )

        print(
            f"\n✅ Destilación por Consejo finalizada en {time.time() - start_time:.2f}s"
        )
        student.save_genomic_model(output_dir)
        print(f"🌟 ORGANISMO COLECTIVO GUARDADO EN: {output_dir}")


if __name__ == "__main__":
    model_path = "/data/data/com.termux/files/home/models/gguf/qwen2-0_5b-q8_0.gguf"
    # DISTILACIÓN QUIRÚRGICA: Solo 2 bloques para validar el enfoque híbrido
    num_test_blocks = 2
    distiller = GenomicDistiller(model_path, num_blocks=num_test_blocks)

    calibration_prompts = [
        "El protocolo GAJE es un sistema de compresión semántica de 2 bits.",
        "La inteligencia artificial en dispositivos móviles requiere eficiencia extrema.",
        "Rust y Python trabajando juntos permiten IA de alto rendimiento.",
        "La compresión genómica preserva la intención del modelo original.",
    ]

    output_dir = "gaje_qwen2_hybrid_v1"
    distiller.run_distillation_pipeline(calibration_prompts, output_dir=output_dir)

    print(f"\n🚀 Destilación Híbrida Completada. Directorio: {output_dir}")
    print(
        "[*] Siguiente paso: Ejecutar benchmarks/distilled_qwen_test.py apuntando a este modelo."
    )
