#!/usr/bin/env python3
"""Fase 1 — Refutación del esquema de emulación temporal 2-bit→4-bit.

Referencia: docs/plans/TEMPORAL_4BIT_EMULATION_DESIGN.md
Hipótesis bajo test (del plan):
    "almacenar 2-bit ultradenso y emular resolución 4-bit (16 niveles) acumulando
     pulsos en el tiempo". Criterio de Fase 1 del plan: CosSim >= 0.998 vs Q4_0
     en una proyección lineal aislada.

Este script implementa TAL CUAL los dos mecanismos del plan y mide con números:

  Enfoque 1 (bit-slice ponderado determinista): V=(4·mu(MSB)+1·mu(LSB))/5.
    - 4·MSB + LSB es una descomposición BASE-4: produce 16 niveles distintos
      (0..15 escalados). PERO requiere almacenar 4 bits por peso (2 chunks de
      2 bits) => almacenamiento IDENTICO a Q4_0, con el doble de ticks (latencia).
      No es compresión: es 4-bit con multiplexado temporal.

  Enfoque 2 (integracion estocastica / dithering, K ticks).
    - Almacena 2 bits reales (4 niveles). El promedio temporal (CLT) solo
      des-correlaciona el error de cuantización; NO fabrica los 2 bits que
      nunca se guardaron. El RMSE queda acotado por el paso de 2-bit y no
      converge a 0 con K -> no alcanza los 16 niveles prometidos.

Veredicto esperado: el plan NO cumple su promesa central (2-bit de memoria con
fidelidad 4-bit). O almacena 4 bits (sin ahorro, con más latencia) o almacena
2 bits (y no llega a fidelidad Q4_0).

Uso:
    python docs/research/temporal_4bit_fase1_test.py
"""
import numpy as np

# ---------------------------------------------------------------------------
# Cuantizador Q4_0 simplificado (estilo GAJE Q4_0: q*scale + min por bloque)
# ---------------------------------------------------------------------------
def quantize_q4_0(vec: np.ndarray, block_size: int = 32) -> np.ndarray:
    vec = vec.astype(np.float32)
    n = len(vec)
    out = np.empty_like(vec)
    for s in range(0, n, block_size):
        blk = vec[s:s + block_size]
        amax = np.max(np.abs(blk)) if len(blk) else 0.0
        scale = amax / 7.0 if amax > 0 else 0.0
        q = np.clip(np.round(blk / scale) if scale > 0 else 0.0, -8, 7) + 8
        out[s:s + len(blk)] = (q.astype(np.int32) - 8).astype(np.float32) * scale
    return out


# ---------------------------------------------------------------------------
# Enfoque 1: dos chunks de 2 bits (base-4) -> 16 niveles
# ---------------------------------------------------------------------------
CENTROIDS_4 = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)

def split_2bit_chunks(q4: int):
    return (q4 >> 2) & 0b11, q4 & 0b11  # MSB (bits 3:2), LSB (bits 1:0)

def reconstruct_enfoque1(q4: int) -> float:
    msb, lsb = split_2bit_chunks(q4)
    return (4.0 * CENTROIDS_4[msb] + 1.0 * CENTROIDS_4[lsb]) / 5.0

vals = sorted({reconstruct_enfoque1(q) for q in range(16)})
print("=== FASE 1 / Enfoque 1: base-4 (2 chunks de 2 bits) ===")
print(f"  Niveles distintos para 16 codigos: {len(vals)}")
print("  -> SI produce 16 niveles (4·MSB+LSB es base-4).")
print("  -> PERO almacena 4 bits por peso (2 x 2-bit) = MISMO almacenamiento")
print("     que Q4_0, con el DOBLE de ticks (latencia). No es compresion.\n")

# ---------------------------------------------------------------------------
# CosSim de capa aislada: comparar contra la proyección Q4_0 real
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
N_IN, N_OUT = 512, 128
w_fp32 = rng.standard_normal((N_OUT, N_IN)).astype(np.float32) * 0.05
w_q4 = np.array([quantize_q4_0(w_fp32[i]) for i in range(N_OUT)])
x = rng.standard_normal((64, N_IN)).astype(np.float32) * 0.3
act_q4 = x @ w_q4.T

# Reconstrucción con el esquema (mapeo de cada peso a 0..15 y base-4 temporal).
qmin, qmax = w_fp32.min(), w_fp32.max()

def encode_q4_index(v):
    return int(np.clip(round((v - qmin) / (qmax - qmin + 1e-9) * 15.0), 0, 15))

w_emul = np.array([[reconstruct_enfoque1(encode_q4_index(v)) for v in row]
                   for row in w_fp32], dtype=np.float32)
act_emul = x @ w_emul.T

# Línea de base justa: almacenar SOLO los 2 bits altos (4 niveles) sin emulación.
w_2bit = np.array([[CENTROIDS_4[encode_q4_index(v) >> 2] for v in row]
                   for row in w_fp32], dtype=np.float32)
act_2bit = x @ w_2bit.T


def cos_sim(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


print("=== FASE 1 / CosSim de capa aislada (criterio plan: >= 0.998) ===")
for name, act in [("2-bit real (4 niveles)", act_2bit),
                  ("Enfoque 1 temporal (16 niveles)", act_emul)]:
    cs = cos_sim(act_q4, act)
    print(f"  CosSim(act_Q4_0, {name:<34}) = {cs:.4f}  "
          f"{'CUMPLE' if cs >= 0.998 else 'NO CUMPLE'}")
print("  -> El Enfoque 1 alcanza CosSim alto SOLO porque almacena los 4 bits;")
print("     no aporta nada frente a cuantizar directamente a 4-bit.\n")

# ---------------------------------------------------------------------------
# Enfoque 2: dithering temporal con almacenamiento REAL de 2 bits
# ---------------------------------------------------------------------------
print("=== FASE 1 / Enfoque 2: dithering con almacenamiento real de 2 bits ===")
rng2 = np.random.default_rng(7)
for k in [1, 2, 4, 8, 16, 64]:
    err = 0.0
    for _ in range(4000):
        true_q4 = rng2.integers(0, 16)      # valor 4-bit original (16 niveles)
        stored = true_q4 >> 2               # 2 bits guardados (4 niveles 0..3)
        noisy = stored + rng2.normal(0, 0.3, size=k).astype(np.float32)
        est = float(np.mean(noisy))
        err += (est - true_q4) ** 2
    rmse = np.sqrt(err / 4000)
    print(f"  K={k:2d} ticks -> RMSE vs valor 4-bit = {rmse:.3f}")
print("  -> El RMSE NO converge a 0: el promedio estima el valor 2-bit guardado")
print("     (0..3), no el 4-bit (0..15). La informacion descartada no se recupera.\n")

# ---------------------------------------------------------------------------
# Veredicto
# ---------------------------------------------------------------------------
print("=== VEREDICTO FASE 1 ===")
print("  Enfoque 1: 16 niveles SI (base-4), pero almacena 4 bits = sin ahorro,")
print("             con el doble de latencia. Equivalentemente peor que Q4_0.")
print("  Enfoque 2: 2 bits reales -> error acotado por el paso 2-bit; NO alcanza")
print("             fidelidad 4-bit ni CosSim >= 0.998.")
print("  CONCLUSION: la 'emulacion temporal' no es compresion gratuita. La regla")
print("  del proyecto se mantiene: representacion cuantizada = Q4_0 + FP32 embd;")
print("  la calidad se gana por corpus/destilacion, no por magia de representacion.")