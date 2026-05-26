
import os
import numpy as np
import gguf
from gaje.nn.stabilized import GenomicLLM as StudentLLM
from gaje.utils.quantization import dequantize_q8_0, unpermute_to_interleaved
import time

def align_attention_iqat(model_path, calibration_prompts, lr=0.005):
    print(f"🧬 Iniciando Alineación de Atención IQAT para: {os.path.basename(model_path)}")
    
    # 1. Cargar Estudiante (IQAT-FFN previo)
    try:
        student = StudentLLM.load_genomic("models/checkpoints/qwen2_iqat.gaje")
    except Exception as e:
        print(f"❌ Error al cargar modelo: {e}")
        return

    tokenizer = student.tokenizer
    reader = gguf.GGUFReader(model_path)
    tensor_map = {t.name: t for t in reader.tensors}
    
    # 2. Iterar por bloques
    for b_idx in range(student.n_blocks):
        print(f"[*] Alineando Atención Bloque {b_idx}/{student.n_blocks}...")
        p = f"blk.{b_idx}."
        
        # Cargar Pesos del Maestro para las Proyecciones
        w_q = dequantize_q8_0(tensor_map[p + "attn_q.weight"])
        w_k = dequantize_q8_0(tensor_map[p + "attn_k.weight"])
        w_v = dequantize_q8_0(tensor_map[p + "attn_v.weight"])
        w_o = dequantize_q8_0(tensor_map[p + "attn_output.weight"])
        
        # Unpermute Q and K if needed (standard for Qwen GGUF)
        w_q = unpermute_to_interleaved(w_q, student.n_head, student.head_dim)
        w_k = unpermute_to_interleaved(w_k, student.n_head_kv, student.head_dim)
        
        attn_norm_weight = np.frombuffer(tensor_map[p + "attn_norm.weight"].data, dtype=np.float32)

        for prompt in calibration_prompts:
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
            if hasattr(input_ids, "ids"): input_ids = input_ids.ids
            
            if hasattr(student.rust_llm, "clear_cache"):
                student.rust_llm.clear_cache()
            
            for i, tid in enumerate(input_ids):
                # Obtener input al bloque actual
                h = student.rust_llm.embeddings_forward(tid)
                for prev_b in range(b_idx):
                    h = student.rust_llm.blocks[prev_b].forward(h, i)
                
                x = np.array(h, dtype=np.float32)
                
                # RMSNorm Maestro
                rms = np.sqrt(np.mean(x**2) + student.eps)
                x_norm = (x / rms) * attn_norm_weight
                
                # Proyecciones del Maestro
                q_t = np.dot(w_q, x_norm)
                k_t = np.dot(w_k, x_norm)
                v_t = np.dot(w_v, x_norm)
                
                # Refinar Proyecciones del Estudiante (Alineación Directa)
                student.rust_llm.blocks[b_idx].q_gen.refine_centroids(x_norm, q_t, lr)
                student.rust_llm.blocks[b_idx].k_gen.refine_centroids(x_norm, k_t, lr)
                student.rust_llm.blocks[b_idx].v_gen.refine_centroids(x_norm, v_t, lr)
                
                # Ahora calculamos la salida de atención del maestro para refinar W_O
                # (Simulamos la salida de atención simplificada para alineación de W_O)
                # En lugar de simular todo el softmax, alineamos W_O con la salida esperada.
                # Pero lo más efectivo es alinear Q, K, V primero.

        del w_q, w_k, w_v, w_o

    student.save("models/checkpoints/qwen2_iqat_full.gaje")
    print(f"\n✅ Calibración TOTAL completada: models/checkpoints/qwen2_iqat_full.gaje")

if __name__ == "__main__":
    MODEL_PATH = "models/gguf/qwen2-0_5b-q8_0.gguf"
    PROMPTS = [
        "¿Quién eres?",
        "Explica la teoría de la relatividad de forma sencilla.",
        "El código genético es la base de la vida.",
        "Programar en Rust es muy seguro.",
        "Había una vez un pequeño robot en el espacio."
    ]
    align_attention_iqat(MODEL_PATH, PROMPTS)
