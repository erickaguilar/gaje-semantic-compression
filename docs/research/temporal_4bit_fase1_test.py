#!/usr/bin/env python3
"""🧬 Test de Validación Fase 1: Emulación Temporal 2-Bit -> 4-Bit vs Q4_0 y 2-Bit Puro.

Implementa la prueba de concepto definida en `docs/plans/TEMPORAL_4BIT_EMULATION_DESIGN.md`:
1. Compara la reconstrucción de pesos en FP32, Q4_0 (16 niveles), 2-Bit Puro (4 niveles)
   y Emulación Temporal 2-Bit -> 4-Bit (en 2 ticks del reloj interno).
2. Simula la propagación a través de 120 capas del transformer para medir si se erradica
   el colapso exponencial (0.97^120 ≈ 0.02) reteniendo CosSim ≥ 0.985.
"""

import numpy as np


def quantize_fp32_to_q4(weights: np.ndarray, block_size: int = 32):
    """Cuantiza a Q4_0 estándar (16 niveles por bloque con scale + min)."""
    orig_shape = weights.shape
    flat = weights.flatten()
    pad_len = (block_size - (len(flat) % block_size)) % block_size
    if pad_len > 0:
        flat = np.pad(flat, (0, pad_len))

    blocks = flat.reshape(-1, block_size)
    b_min = blocks.min(axis=1, keepdims=True)
    b_max = blocks.max(axis=1, keepdims=True)
    scale = (b_max - b_min) / 15.0
    scale[scale == 0] = 1e-7

    q = np.clip(np.round((blocks - b_min) / scale), 0, 15).astype(np.uint8)
    dequant = q * scale + b_min
    return q, dequant.flatten()[: len(flat) - pad_len].reshape(orig_shape)


def quantize_fp32_to_2bit(weights: np.ndarray):
    """Cuantiza a 2-bits genómicos puros (4 centroides globales A, C, G, T)."""
    flat = weights.flatten()
    p25, p50, p75 = np.percentile(flat, [25, 50, 75])
    centroids = np.array([
        flat[flat < p25].mean() if np.any(flat < p25) else -1.0,
        flat[(flat >= p25) & (flat < p50)].mean() if np.any((flat >= p25) & (flat < p50)) else -0.3,
        flat[(flat >= p50) & (flat < p75)].mean() if np.any((flat >= p50) & (flat < p75)) else 0.3,
        flat[flat >= p75].mean() if np.any(flat >= p75) else 1.0,
    ], dtype=np.float32)

    indices = np.zeros(flat.shape, dtype=np.uint8)
    diffs = np.abs(flat[:, None] - centroids[None, :])
    indices = np.argmin(diffs, axis=1).astype(np.uint8)
    dequant = centroids[indices]
    return indices.reshape(weights.shape), dequant.reshape(weights.shape)


def temporal_emulation_2to4(q4_indices: np.ndarray, weights_shape: tuple):
    """Emula la resolución de 4-bits descomponiendo en 2 streams de 2-bits temporales (MSB y LSB).

    Tick t0: Despacho de MSB (2-bits: cuadrante grueso, peso 4x).
    Tick t1: Despacho de LSB (2-bits: ajuste fino, peso 1x).
    Integración en membrana: V_mem = (MSB * 4 + LSB * 1)

    ⚠️ NOTA DE MEMORIA: este esquema ALMACENA los 4 bits (msb_2bit + lsb_2bit).
    `msb*4 + lsb` es una descomposición base-4 que reconstruye el código 4-bit
    completo (0..15). NO es 2-bits por peso: es Q4_0 con los 4 bits entregados
    en 2 ticks. La densidad física es 4 bits/peso, idéntica a Q4_0.
    """
    flat_q = q4_indices.flatten()
    # Descomposición en 2 bits cada uno (SE ALMACENAN AMBOS: 4 bits en total)
    msb_2bit = (flat_q >> 2) & 0b11   # Bits [3..2] (0..3)
    lsb_2bit = flat_q & 0b11          # Bits [1..0] (0..3)

    # Integración temporal en la Timing Wheel (2 ticks)
    reconstructed_q4 = (msb_2bit.astype(np.float32) * 4.0 + lsb_2bit.astype(np.float32) * 1.0)
    return msb_2bit, lsb_2bit, reconstructed_q4.reshape(weights_shape)


def temporal_2bit_real(q4_indices: np.ndarray, weights_shape: tuple):
    """Versión que SÍ reduce memoria a 2 bits/peso: guarda solo el MSB y descarta el LSB.

    Con solo 2 bits se obtienen 4 niveles; `msb` ya no basta para reconstruir los
    16 niveles de Q4_0. Es el régimen en el que el ahorro de RAM es real, y a su
    vez donde la fidelidad 4-bit se pierde irremediablemente.
    """
    flat_q = q4_indices.flatten()
    msb_2bit = (flat_q >> 2) & 0b11   # Único chunk guardado (2 bits)
    return msb_2bit.reshape(weights_shape)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten()
    b_flat = b.flatten()
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


def main():
    print("=" * 75)
    print("🧬 GAJE PROTOCOL: TEST DE EMULACIÓN TEMPORAL 2-BIT -> 4-BIT (FASE 1)")
    print("=" * 75)

    np.random.seed(42)
    dim = 896  # Dimensión de proyección de Qwen2 0.5B
    num_layers = 120  # Número total de proyecciones lineales en el transformer

    print(f"[*] Configuración: Matriz de Proyección [{dim}x{dim}] | Simulación: {num_layers} capas")
    print("-" * 75)

    # 1. Generar matriz de pesos calibrada con distribución normal típica de transformers
    w_fp32 = np.random.randn(dim, dim).astype(np.float32) * (1.0 / np.sqrt(dim))
    x_input = np.random.randn(dim).astype(np.float32)

    # 2. Cuantizaciones
    q4_raw, w_q4 = quantize_fp32_to_q4(w_fp32)
    q2_raw, w_2bit = quantize_fp32_to_2bit(w_fp32)

    # 3. Emulación Temporal (2 Ticks en Timing Wheel)
    msb_tick0, lsb_tick1, q_emulated = temporal_emulation_2to4(q4_raw, w_fp32.shape)

    # 3b. RÉGIMEN REAL DE 2 BITS: guarda solo el MSB (ahorro de memoria real)
    q_2bit_real = temporal_2bit_real(q4_raw, w_fp32.shape)

    # Reconstrucción continua con escala de bloque
    flat = w_fp32.flatten()
    pad_len = (32 - (len(flat) % 32)) % 32
    if pad_len > 0:
        flat = np.pad(flat, (0, pad_len))
    blocks = flat.reshape(-1, 32)
    b_min = blocks.min(axis=1, keepdims=True)
    b_max = blocks.max(axis=1, keepdims=True)
    scale = (b_max - b_min) / 15.0
    scale[scale == 0] = 1e-7
    w_temporal = (q_emulated.reshape(-1, 32) * scale + b_min).flatten()[:len(w_fp32.flatten())].reshape(w_fp32.shape)

    # 4. Evaluación de 1 Capa Aislada
    cossim_q4 = cosine_similarity(w_fp32, w_q4)
    cossim_2bit = cosine_similarity(w_fp32, w_2bit)
    cossim_temporal = cosine_similarity(w_fp32, w_temporal)

    mse_q4 = float(np.mean((w_fp32 - w_q4) ** 2))
    mse_2bit = float(np.mean((w_fp32 - w_2bit) ** 2))
    mse_temporal = float(np.mean((w_fp32 - w_temporal) ** 2))

    print(f"\n📊 1. MÉTRICAS EN CAPA INDIVIDUAL (Proyección Lineal Aislada):")
    print(f"  • FP32 Original           : CosSim = 1.000000 | MSE = 0.000000 | Densidad: 32 bits/peso")
    print(f"  • Q4_0 Estándar (4-bits)  : CosSim = {cossim_q4:.6f} | MSE = {mse_q4:.8f} | Densidad: 4.0 bits/peso")
    print(f"  • 2-Bit Puro (4 estados)  : CosSim = {cossim_2bit:.6f} | MSE = {mse_2bit:.8f} | Densidad: 2.0 bits/peso")
    print(f"  • Emulación Temporal (2t) : CosSim = {cossim_temporal:.6f} | MSE = {mse_temporal:.8f} | Densidad REAL: 4.0 bits/peso (msb+lsb)")

    # 4b. Densidad almacenada real vs fidelidad (el nudo de la hipótesis)
    print(f"  • Emulación Temporal      : IDENTICA a Q4_0 (CosSim/MSE iguales) => NO ahorra memoria,")
    print(f"                              solo añade 2 ticks de latencia.")
    print(f"  • 2-bit real (solo MSB)   : Densidad REAL: 2.0 bits/peso (ahorro SI) pero con 4 niveles,")
    print(f"                              pierde la resolución 4-bit (ver propagación multicapa).")

    # 5. Simulación de Propagación Multicapa Acumulada (120 Capas)
    print(f"\n🌊 2. PROPAGACIÓN MULTICAPA ACUMULADA ({num_layers} Capas Lineales en Cascada):")

    h_fp32 = x_input.copy()
    h_q4 = x_input.copy()
    h_2bit = x_input.copy()
    h_temporal = x_input.copy()
    h_2bit_real = x_input.copy()

    for layer in range(num_layers):
        # Matriz por capa
        w_l = np.random.randn(dim, dim).astype(np.float32) * (1.0 / np.sqrt(dim))
        q4_l, w_q4_l = quantize_fp32_to_q4(w_l)
        _, w_2bit_l = quantize_fp32_to_2bit(w_l)
        _, _, q_em_l = temporal_emulation_2to4(q4_l, w_l.shape)
        q_2r_l = temporal_2bit_real(q4_l, w_l.shape)

        f_l = w_l.flatten()
        pad = (32 - (len(f_l) % 32)) % 32
        if pad > 0:
            f_l = np.pad(f_l, (0, pad))
        blks = f_l.reshape(-1, 32)
        bm = blks.min(axis=1, keepdims=True)
        bM = blks.max(axis=1, keepdims=True)
        sc = (bM - bm) / 15.0
        sc[sc == 0] = 1e-7
        w_temp_l = (q_em_l.reshape(-1, 32) * sc + bm).flatten()[:len(w_l.flatten())].reshape(w_l.shape)
        w_2r_l = (q_2r_l.reshape(-1, 32) * (sc * 4.0) + bm).flatten()[:len(w_l.flatten())].reshape(w_l.shape)

        h_fp32 = np.dot(h_fp32, w_l)
        h_q4 = np.dot(h_q4, w_q4_l)
        h_2bit = np.dot(h_2bit, w_2bit_l)
        h_temporal = np.dot(h_temporal, w_temp_l)
        h_2bit_real = np.dot(h_2bit_real, w_2r_l)

        # Normalización de capa para estabilidad numérica
        h_fp32 /= (np.linalg.norm(h_fp32) + 1e-6)
        h_q4 /= (np.linalg.norm(h_q4) + 1e-6)
        h_2bit /= (np.linalg.norm(h_2bit) + 1e-6)
        h_temporal /= (np.linalg.norm(h_temporal) + 1e-6)
        h_2bit_real /= (np.linalg.norm(h_2bit_real) + 1e-6)

    cossim_e2e_q4 = cosine_similarity(h_fp32, h_q4)
    cossim_e2e_2bit = cosine_similarity(h_fp32, h_2bit)
    cossim_e2e_temporal = cosine_similarity(h_fp32, h_temporal)
    cossim_e2e_2bit_real = cosine_similarity(h_fp32, h_2bit_real)

    print(f"  • Q4_0 Estándar Final (120 capas)  : CosSim = {cossim_e2e_q4:.6f} {'✅ RETENCIÓN' if cossim_e2e_q4 > 0.90 else '🔴 DEGRADADO'}")
    print(f"  • 2-Bit Puro Final (120 capas)     : CosSim = {cossim_e2e_2bit:.6f} {'🔴 COLAPSO SEMÁNTICO (Ruido)' if cossim_e2e_2bit < 0.10 else '🟡 PARCIAL'}")
    print(f"  • Emulación Temporal Final (120 c) : CosSim = {cossim_e2e_temporal:.6f} {'🏆 PARIDAD PERFECTA A Q4' if cossim_e2e_temporal >= cossim_e2e_q4 - 0.01 else '🟡 MEJORA'}  [4 bits/peso]")
    print(f"  • 2-bit real Final (solo MSB,120)  : CosSim = {cossim_e2e_2bit_real:.6f} {'🔴 COLAPSO' if cossim_e2e_2bit_real < 0.10 else '🟡 PARCIAL'}  [2 bits/peso: ahorro REAL]")

    print("\n" + "=" * 75)
    print("🏆 CONCLUSIÓN DEL TEST DE FASE 1:")
    print("  ⚠️ La 'Emulación Temporal' NO reduce memoria: almacena 4 bits/peso (msb+lsb)")
    print("     y es bit a bit IDÉNTICA a Q4_0. Su etiqueta '2.0 bits/peso' era un error.")
    print("  🔴 El único régimen que reduce a 2 bits/peso reales (guardar solo MSB) pierde")
    print("     la resolución 4-bit -> colapso semántico en la cascada (CosSim -> 0).")
    print("  ✅ VEREDICTO: no existe '2-bit de memoria con fidelidad 4-bit'. Ahorrar RAM")
    print("     y conservar Q4_0 son mutuamente excluyentes; la temporalidad solo añade")
    print("     latencia (2 ticks) al costo de no ahorrar nada.")
    print("=" * 75)


if __name__ == "__main__":
    main()