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
    # Si el modelo es muy grande para Termux, StudentLLM podría fallar por OOM.
    # SmolLM2-135M es la opción recomendada para dispositivos con < 4GB RAM.
    try:
        student = StudentLLM(model_path)
    except Exception as e:
        print(f"❌ Error crítico al cargar el modelo: {e}")
        print("💡 Sugerencia: Si estás en Termux, intenta usar smollm2-135m-q8_0.gguf")
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
        ffn_norm_weight = tensor_map[p + "ffn_norm.weight"].data.astype(np.float32)

        for prompt in calibration_prompts:
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)

            # Reset cache para cada prompt (asegura independencia de contexto)
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
                # El estudiante ajusta sus centroides para minimizar la deriva (drift)
                student.blocks[b_idx].refine_ffn(x_norm, teacher_swiglu_target, lr)

        # Liberar memoria de los pesos del maestro del bloque actual
        del w_gate_f32
        del w_up_f32

    print("\n✅ Calibración completada con éxito.")


if __name__ == "__main__":
    # Prioridad: SmolLM para estabilidad en Termux, luego Qwen
    models = [
        "/data/data/com.termux/files/home/models/gguf/smollm2-135m-q8_0.gguf",
        "/data/data/com.termux/files/home/models/gguf/qwen2-0_5b-q8_0.gguf",
    ]

    MODEL_PATH = next((m for m in models if os.path.exists(m)), None)

    if not MODEL_PATH:
        print("❌ Error: No se encontró ningún modelo GGUF compatible.")
        print(f"Buscado en: {models}")
    else:
        PROMPTS = [
            "¿Cuál es la capital de México?",
            "Explica el protocolo GAJE de compresión genómica.",
            "The stars twinkle in the night sky.",
        ]

        apply_iqat_calibration(MODEL_PATH, PROMPTS)
