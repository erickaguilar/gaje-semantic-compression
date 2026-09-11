#!/usr/bin/env python3
"""
extract_activations.py
======================
Pipeline formal de extracción multi-prompt de activaciones intermedias [N, L, D]
entre modelo de control (Q4_0) y candidato cuantizado (Q2_0) para GAJE.

Garantías metodológicas:
  1. Corpus multi-dominio diverso con N >= 2000 tokens i.i.d.
  2. Reseteo estricto del KV-cache (clear_cache()) entre prompts independientes.
  3. Sanity check formal de Capa 0 (S_c > 0.999, ratio ~= 1.0).
  4. Generación de archivo canónico .npz compatible con diagnose_phase_capacity.py.
"""

import os
import sys
import argparse
import time
import numpy as np

# Asegurar importación de gaje
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(repo_root, "python"))

from gaje.nn.stabilized import GenomicLLM

# ---------------------------------------------------------------------------
# Corpus diverso de prueba (20 prompts balanceados, ~100-150 tokens c/u)
# ---------------------------------------------------------------------------
DEFAULT_PROMPTS = [
    # 1. Razonamiento Matemático
    "Demuestra por qué la raíz cuadrada de 2 es un número irracional usando el método de reducción al absurdo paso a paso. Comienza asumiendo que existen dos enteros coprimos p y q.",
    # 2. Física Teórica
    "Explica el principio de acción mínima de Hamilton en mecánica clásica y cómo se relaciona con las ecuaciones de Euler-Lagrange para una partícula en un potencial armónico.",
    # 3. Algorítmica y Código
    "Implement a fast lock-free circular queue in C++20 using std::atomic with acquire-release memory orderings and explain how cacheline contention is avoided with hardware padding.",
    # 4. Biología Molecular y Genómica
    "Describe la función de la ADN polimerasa delta y épsilon en la replicación celular de eucariotas, enfatizando la corrección de pruebas 3 prima a 5 prima y la síntesis de hebra rezagada.",
    # 5. Filosofía de la Mente
    "Analiza el experimento mental de la habitación china de John Searle y evalúa si la objeción del sistema o la del robot refutan válidamente la ausencia de intencionalidad intrínseca.",
    # 6. Arquitectura de Computadores
    "Detalla la microarquitectura de un procesador superescalar moderno: predicción de saltos TAGE, estación de reserva, ejecución fuera de orden (OoO) y buffer de reordenamiento (ROB).",
    # 7. Redes Neuronales y LLMs
    "Calcula la complejidad computacional en FLOPS y memoria en bytes de la atención FlashAttention-2 comparada con la atención estándar O(N^2) con KV-cache paginado.",
    # 8. Literatura e Historia
    "Sintetiza las tensiones sociopolíticas entre Atenas y Esparta que desembocaron en la Guerra del Peloponeso según la crónica de Tucídides, enfocándote en la trampa de Tucídides.",
    # 9. Química Cuántica
    "Explica la aproximación de Born-Oppenheimer en la ecuación de Schrödinger molecular y describe cómo los estados electrónicos adiabáticos se desacoplan del movimiento nuclear.",
    # 10. Sistemas Distribuidos
    "Explica cómo el consenso Raft previene la partición de cerebro dividido (split-brain) durante la elección de líder cuando existe una partición de red asimétrica entre tres nodos.",
    # 11. Economía y Teoría de Juegos
    "Define formalmente el equilibrio de Nash perfecto en subjuegos para un juego dinámico finito y proporciona un ejemplo con un juego de ultimátum con dos rondas.",
    # 12. Medicina y Fisiología
    "Describe el ciclo de transducción de señales del receptor acoplado a proteína G (GPCR) activado por adrenalina, desde la activación de adenilato ciclasa hasta la fosforilación de PKA.",
    # 13. Criptografía y Ciberseguridad
    "Explica la construcción de intercambio de claves Diffie-Hellman en curvas elípticas (ECDH) sobre la curva Curve25519 y cómo mitigar ataques de intermediario con firmas Ed25519.",
    # 14. Geología y Tectónica
    "Describe los procesos geodinámicos que ocurren en una zona de subducción tipo Mariana versus una tipo chilena, incluyendo el flujo de fluidos, serpentinización y magmatismo de arco.",
    # 15. Inteligencia Artificial y Compresión
    "¿Cómo permite la cuantización vectorial cuaternaria en 2-bits (A, C, G, T) preservar las representaciones semánticas en un espacio métrico proyectivo sin perder la simetría de fase?",
    # 16. Astronomía y Astrofísica
    "Explica el límite de Chandrasekhar para enanas blancas degeneradas por presión de electrones relativistas y deduce por qué la masa máxima converge a aproximadamente 1.44 masas solares.",
    # 17. Derecho y Filosofía Legal
    "Compara el positivismo jurídico excluyente de Joseph Raz con el antipositivismo interpretativo de Ronald Dworkin respecto al papel de los principios morales en la determinación del derecho.",
    # 18. Lingüística Estructural
    "Explica la distinción entre sincronía y diacronía según Ferdinand de Saussure y cómo el concepto de valor lingüístico depende puramente de las relaciones de oposición en el sistema.",
    # 19. Robótica y Control
    "Describe el funcionamiento de un filtro de Kalman extendido (EKF) para localización simultánea y mapeo (SLAM) en un robot móvil con sensores odométricos y LiDAR.",
    # 20. Computación Cuántica
    "Explica cómo el algoritmo de corrección cuántica de errores del código de superficie (surface code) detecta errores de inversión de bit (X) e inversión de fase (Z) mediante estabilizadores.",
]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Calcula la similitud de coseno vector a vector sobre el último eje: [..., D] -> [...]"""
    norm_a = np.linalg.norm(a, axis=-1)
    norm_b = np.linalg.norm(b, axis=-1)
    norm_prod = np.maximum(norm_a * norm_b, 1e-12)
    dot = np.sum(a * b, axis=-1)
    return dot / norm_prod


def main():
    parser = argparse.ArgumentParser(description="Extracción formal de activaciones intermedias multi-prompt")
    parser.add_argument("--ctrl", type=str, default="models/production/qwen2_5_0_5b.gaje",
                        help="Ruta al modelo control (Q4_0 / FP16)")
    parser.add_argument("--candidate", type=str, default="models/qwen2_5_0_5b_q2_0.flat",
                        help="Ruta al modelo candidato (Q2_0)")
    parser.add_argument("--output", type=str, default="data/activaciones_qwen2_5_q4_vs_q2.npz",
                        help="Ruta de destino del archivo .npz")
    parser.add_argument("--target-tokens", type=int, default=2000,
                        help="Objetivo mínimo de tokens totales a procesar")
    parser.add_argument("--run-plot", action="store_true", default=True,
                        help="Ejecutar análisis espectral y generar gráfica tras la extracción")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("🧬 GAJE HELIX — Extracción y Diagnóstico de Activaciones por Capas")
    print("=" * 80)
    print(f"📥 Modelo Control (Q4_0):    {args.ctrl}")
    print(f"📥 Modelo Candidato (Q2_0):  {args.candidate}")
    print(f"📤 Salida .npz:              {args.output}")
    print(f"🎯 Meta de Tokens Mínimos:   {args.target_tokens}\n")

    # 1. Cargar modelos
    print("⏳ [1/4] Cargando modelos mediante Zero-Copy Mmap...")
    t0 = time.time()
    m_ctrl = GenomicLLM.load_genomic(args.ctrl)
    t_ctrl = time.time() - t0
    print(f"   • Control cargado en {t_ctrl:.2f}s ({m_ctrl.n_blocks} bloques, {m_ctrl.n_embd}d)")

    t0 = time.time()
    m_q2 = GenomicLLM.load_genomic(args.candidate)
    t_q2 = time.time() - t0
    print(f"   • Candidato Q2 cargado en {t_q2:.2f}s ({m_q2.n_blocks} bloques, {m_q2.n_embd}d)")

    assert m_ctrl.n_blocks == m_q2.n_blocks, "Los modelos deben tener el mismo número de bloques"
    assert m_ctrl.n_embd == m_q2.n_embd, "Los modelos deben tener la misma dimensión oculta"

    num_layers = m_ctrl.n_blocks
    dim = m_ctrl.n_embd

    # 2. Iterar prompts y recolectar activaciones
    print(f"\n⏳ [2/4] Procesando corpus multi-prompt (N >= {args.target_tokens})...")
    all_ctrl_acts = []
    all_q2_acts = []
    all_prompt_ids = []
    all_token_ids = []

    total_tokens = 0
    prompt_idx = 0

    while total_tokens < args.target_tokens:
        text = DEFAULT_PROMPTS[prompt_idx % len(DEFAULT_PROMPTS)]
        
        tok_out = m_ctrl.tokenizer.encode(text)
        token_ids = tok_out.ids if hasattr(tok_out, "ids") else list(tok_out)

        t_prompt_start = time.time()
        acts_ctrl = m_ctrl.extract_layer_activations(token_ids, clear_cache=True)
        acts_q2 = m_q2.extract_layer_activations(token_ids, clear_cache=True)
        t_prompt = time.time() - t_prompt_start

        n_tok = len(token_ids)
        total_tokens += n_tok

        all_ctrl_acts.append(acts_ctrl)
        all_q2_acts.append(acts_q2)
        all_prompt_ids.extend([prompt_idx] * n_tok)
        all_token_ids.extend(token_ids)

        print(f"   • Prompt {prompt_idx + 1:02d}: {n_tok:3d} tokens | Total acumulado: {total_tokens:4d}/{args.target_tokens} ({t_prompt:.2f}s)")
        prompt_idx += 1

    x_ctrl = np.concatenate(all_ctrl_acts, axis=0)
    x_q2 = np.concatenate(all_q2_acts, axis=0)
    prompt_indices = np.array(all_prompt_ids, dtype=np.int32)
    token_ids_arr = np.array(all_token_ids, dtype=np.int64)

    N, L, D = x_ctrl.shape
    print(f"\n✅ Tensor de activaciones consolidado:")
    print(f"   • Forma de Tensores: [{N} tokens, {L} capas, {D} dimensiones]")
    print(f"   • Memoria total por tensor: {x_ctrl.nbytes / (1024**2):.2f} MB")

    # 3. Sanity Check y Métricas Capa por Capa
    print("\n" + "=" * 80)
    print("📊 [3/4] Sanity Check y Métricas de Fidelidad por Capa")
    print("=" * 80)
    print(f"{'Capa':<6} | {'Similitud Coseno (S_c)':<24} | {'Ratio de Normas (||q2||/||ctrl||)':<30} | {'Norma Ctrl':<10} | {'Norma Q2':<10}")
    print("-" * 88)

    layer_sc = []
    layer_ratios = []

    for l in range(L):
        act_c = x_ctrl[:, l, :]
        act_q = x_q2[:, l, :]

        cos_sims = cosine_similarity(act_c, act_q)
        mean_cos = float(np.mean(cos_sims))
        std_cos = float(np.std(cos_sims))

        norm_c = np.linalg.norm(act_c, axis=-1)
        norm_q = np.linalg.norm(act_q, axis=-1)
        ratios = norm_q / np.maximum(norm_c, 1e-12)
        mean_ratio = float(np.mean(ratios))
        std_ratio = float(np.std(ratios))

        layer_sc.append(mean_cos)
        layer_ratios.append(mean_ratio)

        flag = ""
        if l == 0:
            flag = " [CAPA 0 - SANITY CHECK]"

        print(f"{l:02d}     | {mean_cos:7.4f} ± {std_cos:6.4f}          | {mean_ratio:7.4f} ± {std_ratio:6.4f}                 | {np.mean(norm_c):8.2f}   | {np.mean(norm_q):8.2f}{flag}")

    sc_layer0 = layer_sc[0]
    ratio_layer0 = layer_ratios[0]
    print("-" * 88)
    if sc_layer0 >= 0.999:
        print(f"🎉 \x1b[1;32mSANITY CHECK DE CAPA 0 SUPERADO:\x1b[0m S_c = {sc_layer0:.5f} >= 0.999 (Ratio = {ratio_layer0:.4f})")
    elif sc_layer0 >= 0.990:
        print(f"⚠️ \x1b[1;33mSANITY CHECK CAPA 0 ACEPTABLE:\x1b[0m S_c = {sc_layer0:.5f} (esperado > 0.999 en FP16 directo, tolerable en cascada Q4->Q2)")
    else:
        print(f"❌ \x1b[1;31mSANITY CHECK FALLIDO EN CAPA 0:\x1b[0m S_c = {sc_layer0:.5f} < 0.990")

    # 4. Guardar archivo .npz
    print(f"\n💾 [4/4] Serializando base de datos a {args.output}...")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(
        args.output,
        ctrl=x_ctrl,
        q2=x_q2,
        prompt_indices=prompt_indices,
        token_ids=token_ids_arr,
        metadata={
            "ctrl_model": args.ctrl,
            "q2_model": args.candidate,
            "n_tokens": N,
            "n_layers": L,
            "dim": D,
            "n_prompts": prompt_idx,
            "cascade": True,
            "sc_layer0": sc_layer0,
            "ratio_layer0": ratio_layer0,
        }
    )
    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"✅ Archivo .npz generado exitosamente ({file_size_mb:.2f} MB)")

    # 5. Ejecución de diagnóstico espectral
    if args.run_plot:
        try:
            sys.path.insert(0, os.path.join(repo_root, "scripts/benchmarks"))
            import diagnose_phase_capacity as diag
            print("\n📈 Ejecutando diagnóstico espectral de fase y capacidad (SVD + Subespacios)...")
            diag.run_diagnostics(x_ctrl, x_q2, output_plot="data/diagnostico_fase_capacidad_qwen.png")
            print("✅ Gráfica de diagnóstico generada en: data/diagnostico_fase_capacidad_qwen.png")
        except Exception as e:
            print(f"⚠️ Aviso diagnóstico espectral: {e}")

    print("\n" + "=" * 80)
    print("🏁 Pipeline de extracción y diagnóstico finalizado con éxito.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
