import os
import numpy as np
import gguf
from gaje.nn.stabilized import GenomicLLM as StudentLLM
from gaje.utils.quantization import dequantize_q8_0


def apply_iqat_calibration(model_path, calibration_prompts, lr=0.005):
    print(
        f"🚀 Iniciando Calibración IQAT Quirúrgica para: {os.path.basename(model_path)}"
    )

    # 1. Cargar Estudiante (2-bit)
    try:
        student = StudentLLM.load_genomic("models/checkpoints/qwen2_base.gaje")
    except Exception as e:
        print(f"❌ Error crítico al cargar el modelo estudiante: {e}")
        return

    tokenizer = student.tokenizer
    reader = gguf.GGUFReader(model_path)

    # Mapeo de tensores para carga rápida del Maestro
    tensor_map = {t.name: t for t in reader.tensors}

    # 2. Iterar por bloques
    for b_idx in range(student.n_blocks):
        print(f"\n[*] Calibrando Bloque {b_idx}/{student.n_blocks}...")
        p = f"blk.{b_idx}."

        # Verificar si es una capa FFN (Qwen/SmolLM style)
        if p + "ffn_gate.weight" not in tensor_map:
            print(
                f"      [!] Bloque {b_idx} no tiene pesos SwiGLU estándar. Saltando refinamiento FFN..."
            )
            continue

        # Cargar Pesos del Maestro (Teacher) solo para este bloque FFN
        w_gate_f32 = dequantize_q8_0(tensor_map[p + "ffn_gate.weight"])
        w_up_f32 = dequantize_q8_0(tensor_map[p + "ffn_up.weight"])
        ffn_norm_weight = np.frombuffer(
            tensor_map[p + "ffn_norm.weight"].data, dtype=np.float32
        )

        for prompt in calibration_prompts:
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
            if hasattr(input_ids, "ids"):
                input_ids = input_ids.ids

            # Reset cache para cada prompt
            if hasattr(student.rust_llm, "clear_cache"):
                student.rust_llm.clear_cache()

            # Capturar activaciones fluyendo por el estudiante
            for i, tid in enumerate(input_ids):
                # 1. Correr forward hasta el bloque actual para obtener el input
                h = student.rust_llm.embeddings_forward(tid)
                for prev_b in range(b_idx):
                    h = student.rust_llm.blocks[prev_b].forward(h, i)

                # Ahora h es el input al bloque b_idx
                x = np.array(h, dtype=np.float32)

                # 2. Simular el bloque FFN del Maestro (Teacher)
                # RMSNorm del maestro
                rms = np.sqrt(np.mean(x**2) + student.eps)
                x_norm = (x / rms) * ffn_norm_weight

                # SwiGLU del maestro
                gate = np.dot(w_gate_f32, x_norm)
                up = np.dot(w_up_f32, x_norm)
                # SiLU
                silu_gate = gate * (1.0 / (1.0 + np.exp(-np.clip(gate, -20, 20))))
                teacher_swiglu_target = silu_gate * up

                # 3. Refinamiento IQAT Nativo
                student.rust_llm.blocks[b_idx].refine_ffn(
                    x_norm, teacher_swiglu_target, lr
                )

        # Liberar memoria
        del w_gate_f32
        del w_up_f32

    student.save("models/checkpoints/qwen2_iqat.gaje")
    print("\n✅ Calibración IQAT completada: models/checkpoints/qwen2_iqat.gaje")


if __name__ == "__main__":
    MODEL_PATH = "models/gguf/qwen2-0_5b-q8_0.gguf"

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: No se encontró el modelo {MODEL_PATH}")
    else:
        PROMPTS = [
            "Hola, ¿cómo estás?",
            "La inteligencia artificial es una herramienta poderosa.",
            "GAJE protocol allows for 16x compression.",
            "Explica qué es el ADN.",
            "Canta una canción sobre el espacio.",
        ]

        apply_iqat_calibration(MODEL_PATH, PROMPTS)
