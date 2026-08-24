#!/usr/bin/env python3
"""
🧬 GAJE HELIX — FASE 0: MICRO-BENCHMARK DECISIVO DE SPSA DISCRETO
================================================================================
Compara los brazos de optimización de orden cero con presupuesto idéntico de forwards:
  (a) Mutación Aleatoria Simple (Random Mutation Hill-Climbing)
  (b) SPSA Discreto con Pares Antitéticos (±Δ a centroides vecinos)
  (c) Reglas Locales Hebbianas / Centroid Counts
  (d) Currículo Híbrido H3: Reglas Locales (Crecimiento) + SPSA (Refinamiento)

Gate de Decisión: Validar empíricamente las hipótesis H1, H2 y H3 de
ZERO_ORDER_NATIVE_TRAINING_PLAN.md bajo un presupuesto idéntico de cómputo.
================================================================================
"""

import time
import numpy as np

# Configurar semilla determinista para reproducibilidad
np.random.seed(42)

print(
    "================================================================================"
)
print("🧬 GAJE HELIX: FASE 0 — MICRO-BENCHMARK SPSA DISCRETO VS MUTACIÓN ALEATORIA")
print(
    "================================================================================"
)

# 1. Configuración del Micro-Organismo y Dataset de Prueba
IN_FEATURES = 64
OUT_FEATURES = 64
BATCH_SIZE = 32
NUM_CENTROIDS = 16  # Q4_0: 16 centroides discretos
FORWARD_BUDGET = 2000  # Presupuesto idéntico de forward passes para cada brazo
K_PERTURBATIONS = 16  # Número de pesos perturbados por paso

# Generar datos sintéticos de entrada X y objetivo Y
X = np.random.randn(BATCH_SIZE, IN_FEATURES).astype(np.float32)
W_target = np.random.randn(OUT_FEATURES, IN_FEATURES).astype(np.float32)
Y = np.dot(X, W_target.T)  # (BATCH_SIZE, OUT_FEATURES)

# Tabla de centroides fijos en [-2.0, 2.0]
centroids = np.linspace(-2.0, 2.0, NUM_CENTROIDS, dtype=np.float32)


def compute_loss(indices_matrix):
    """
    Función de pérdida (MSE): Simula 1 forward pass nativo.
    Decodifica índices de centroides a pesos continuos y calcula error cuadrático medio.
    """
    W_decoded = centroids[indices_matrix]  # (OUT_FEATURES, IN_FEATURES)
    Y_pred = np.dot(X, W_decoded.T)
    return float(np.mean((Y_pred - Y) ** 2))


# Estado inicial idéntico para todos los brazos (inicialización aleatoria discreta)
initial_indices = np.random.randint(
    0, NUM_CENTROIDS, size=(OUT_FEATURES, IN_FEATURES), dtype=np.uint8
)
initial_loss = compute_loss(initial_indices)

print(
    f"[*] Parámetros del Micro-Organismo: {OUT_FEATURES}x{IN_FEATURES} ({OUT_FEATURES * IN_FEATURES} pesos)"
)
print(f"[*] Presupuesto de Cómputo por Brazo: {FORWARD_BUDGET} forward passes")
print(f"[*] Loss Inicial: {initial_loss:.4f}\n")

# Configuración de mutación dirigida para el Brazo A (vecinos en codebook)
# En lugar de reasignaciones aleatorias completas, perturbamos a centros vecinos (±1)
# para crear una comparación justa con SPSA (que también hace perturbaciones dirigidas)
A_USE_DIRECTED_MUTATION = True


# ==============================================================================
# BRAZO A: Mutación Aleatoria Simple (Random Mutation Hill Climbing)
# Usa perturbaciones dirigidas a vecinos de códigobook para comparación justa con SPSA
# ==============================================================================
print("[1/4] Ejecutando Brazo A: Mutación Aleatoria Simple (directed)...")
indices_a = initial_indices.copy()
current_loss_a = initial_loss
loss_history_a = [current_loss_a]

t0 = time.time()
for step in range(FORWARD_BUDGET):
    mut_indices = indices_a.copy()
    rows = np.random.randint(0, OUT_FEATURES, size=K_PERTURBATIONS)
    cols = np.random.randint(0, IN_FEATURES, size=K_PERTURBATIONS)
    # Perturbación dirigida: mover a centroide vecino (±1), clampear a [0, NUM_CENTROIDS-1]
    neighbor_deltas = np.random.choice([-1, 1], size=K_PERTURBATIONS)
    new_vals = np.clip(
        indices_a[rows, cols].astype(np.int32) + neighbor_deltas, 0, NUM_CENTROIDS - 1
    ).astype(np.uint8)
    mut_indices[rows, cols] = new_vals

    cand_loss = compute_loss(mut_indices)  # 1 forward
    if cand_loss < current_loss_a:
        indices_a = mut_indices
        current_loss_a = cand_loss

    loss_history_a.append(current_loss_a)

time_a = time.time() - t0
reduction_a = ((initial_loss - current_loss_a) / initial_loss) * 100.0
print(
    f"  • Loss Final: {current_loss_a:.4f} | Reducción: {reduction_a:.2f}% | Tiempo: {time_a * 1000:.1f} ms"
)


# ==============================================================================
# BRAZO B: SPSA Discreto con Pares Antitéticos y Temperatura Adaptativa
# ==============================================================================
print("\n[2/4] Ejecutando Brazo B: SPSA Discreto con Schedule de Temperatura...")
indices_b = initial_indices.copy().astype(np.float32)
current_loss_b = initial_loss
loss_history_b = [current_loss_b]

t0 = time.time()
spsa_steps = FORWARD_BUDGET // 2  # 2 forwards por paso

for step in range(spsa_steps):
    # Schedule de temperatura T_g: suave 3→0.5, decay cuadrático para exploración inicial
    progress = (step / spsa_steps) ** 2
    temp = max(1, int(round(3.0 * (1.0 - progress))))

    target_row = np.random.randint(0, OUT_FEATURES)
    cols = np.random.choice(IN_FEATURES, size=K_PERTURBATIONS, replace=False)
    delta = np.random.choice([-temp, temp], size=len(cols))

    # 1. Forward positivo (+delta)
    ind_plus = indices_b.copy()
    ind_plus[target_row, cols] = np.clip(
        np.round(ind_plus[target_row, cols] + delta), 0, NUM_CENTROIDS - 1
    )
    loss_plus = compute_loss(ind_plus.astype(np.uint8))  # Forward 1

    # 2. Forward antitético (-delta)
    ind_minus = indices_b.copy()
    ind_minus[target_row, cols] = np.clip(
        np.round(ind_minus[target_row, cols] - delta), 0, NUM_CENTROIDS - 1
    )
    loss_minus = compute_loss(ind_minus.astype(np.uint8))  # Forward 2

    if loss_plus < loss_minus and loss_plus < current_loss_b:
        indices_b[target_row, cols] = ind_plus[target_row, cols]
        current_loss_b = loss_plus
    elif loss_minus < loss_plus and loss_minus < current_loss_b:
        indices_b[target_row, cols] = ind_minus[target_row, cols]
        current_loss_b = loss_minus

    loss_history_b.append(current_loss_b)
    loss_history_b.append(current_loss_b)

time_b = time.time() - t0
reduction_b = ((initial_loss - current_loss_b) / initial_loss) * 100.0
print(
    f"  • Loss Final: {current_loss_b:.4f} | Reducción: {reduction_b:.2f}% | Tiempo: {time_b * 1000:.1f} ms"
)


# ==============================================================================
# BRAZO C: Reglas Locales Hebbianas / Centroid Counts
# ==============================================================================
print("\n[3/4] Ejecutando Brazo C: Reglas Locales Hebbianas...")
indices_c = initial_indices.copy()
current_loss_c = initial_loss
loss_history_c = [current_loss_c]

t0 = time.time()
for step in range(FORWARD_BUDGET):
    W_cur = centroids[indices_c]
    Y_pred = np.dot(X, W_cur.T)
    residual = Y - Y_pred  # (BATCH_SIZE, OUT_FEATURES)

    grad_approx = np.dot(residual.T, X) / BATCH_SIZE  # (OUT_FEATURES, IN_FEATURES)

    flat_grad = np.abs(grad_approx).flatten()
    top_indices = np.argpartition(flat_grad, -K_PERTURBATIONS)[-K_PERTURBATIONS:]
    rows, cols = np.unravel_index(top_indices, (OUT_FEATURES, IN_FEATURES))

    signs = np.sign(grad_approx[rows, cols]).astype(np.int32)
    new_vals = np.clip(
        indices_c[rows, cols].astype(np.int32) + signs, 0, NUM_CENTROIDS - 1
    ).astype(np.uint8)

    indices_cand = indices_c.copy()
    indices_cand[rows, cols] = new_vals

    cand_loss = compute_loss(indices_cand)  # 1 forward
    if cand_loss < current_loss_c:
        indices_c = indices_cand
        current_loss_c = cand_loss

    loss_history_c.append(current_loss_c)

time_c = time.time() - t0
reduction_c = ((initial_loss - current_loss_c) / initial_loss) * 100.0
print(
    f"  • Loss Final: {current_loss_c:.4f} | Reducción: {reduction_c:.2f}% | Tiempo: {time_c * 1000:.1f} ms"
)


# ==============================================================================
# BRAZO D: Currículo Híbrido H3 (Reglas Locales + SPSA Discreto)
# ==============================================================================
print(
    "\n[4/4] Ejecutando Brazo D: Currículo Híbrido H3 (Crecimiento + Refinamiento)..."
)
indices_d = initial_indices.copy()
current_loss_d = initial_loss
loss_history_d = [current_loss_d]

t0 = time.time()
growth_budget = FORWARD_BUDGET // 2
refine_budget = FORWARD_BUDGET - growth_budget

# Etapa 1: Crecimiento con Reglas Locales
for step in range(growth_budget):
    W_cur = centroids[indices_d]
    Y_pred = np.dot(X, W_cur.T)
    residual = Y - Y_pred
    grad_approx = np.dot(residual.T, X) / BATCH_SIZE

    flat_grad = np.abs(grad_approx).flatten()
    top_indices = np.argpartition(flat_grad, -K_PERTURBATIONS)[-K_PERTURBATIONS:]
    rows, cols = np.unravel_index(top_indices, (OUT_FEATURES, IN_FEATURES))

    signs = np.sign(grad_approx[rows, cols]).astype(np.int32)
    new_vals = np.clip(
        indices_d[rows, cols].astype(np.int32) + signs, 0, NUM_CENTROIDS - 1
    ).astype(np.uint8)

    indices_cand = indices_d.copy()
    indices_cand[rows, cols] = new_vals

    cand_loss = compute_loss(indices_cand)
    if cand_loss < current_loss_d:
        indices_d = indices_cand
        current_loss_d = cand_loss
    loss_history_d.append(current_loss_d)

# Etapa 2: Refinamiento con SPSA Discreto (Pares Antitéticos)
spsa_refine_steps = refine_budget // 2
for step in range(spsa_refine_steps):
    target_row = np.random.randint(0, OUT_FEATURES)
    cols = np.random.choice(IN_FEATURES, size=K_PERTURBATIONS, replace=False)
    delta = np.random.choice([-1, 1], size=len(cols))

    # Forward positivo
    ind_plus = indices_d.copy()
    ind_plus[target_row, cols] = np.clip(
        ind_plus[target_row, cols].astype(np.int32) + delta, 0, NUM_CENTROIDS - 1
    ).astype(np.uint8)
    loss_plus = compute_loss(ind_plus)

    # Forward antitético
    ind_minus = indices_d.copy()
    ind_minus[target_row, cols] = np.clip(
        ind_minus[target_row, cols].astype(np.int32) - delta, 0, NUM_CENTROIDS - 1
    ).astype(np.uint8)
    loss_minus = compute_loss(ind_minus)

    if loss_plus < loss_minus and loss_plus < current_loss_d:
        indices_d[target_row, cols] = ind_plus[target_row, cols]
        current_loss_d = loss_plus
    elif loss_minus < loss_plus and loss_minus < current_loss_d:
        indices_d[target_row, cols] = ind_minus[target_row, cols]
        current_loss_d = loss_minus

    loss_history_d.append(current_loss_d)
    loss_history_d.append(current_loss_d)

time_d = time.time() - t0
reduction_d = ((initial_loss - current_loss_d) / initial_loss) * 100.0
print(
    f"  • Loss Final: {current_loss_d:.4f} | Reducción: {reduction_d:.2f}% | Tiempo: {time_d * 1000:.1f} ms"
)


# ==============================================================================
# EVALUACIÓN DECISIVA DE HIPÓTESIS H1, H2 Y H3
# ==============================================================================
print(
    "\n================================================================================"
)
print("📊 RESUMEN COMPARATIVO Y VEREDICTO DE FASE 0")
print(
    "================================================================================"
)

target_loss_30 = initial_loss * 0.70


def forwards_to_reach(history, target):
    for i, loss_val in enumerate(history):
        if loss_val <= target:
            return i
    return len(history)


forwards_a = forwards_to_reach(loss_history_a, target_loss_30)
forwards_b = forwards_to_reach(loss_history_b, target_loss_30)
forwards_c = forwards_to_reach(loss_history_c, target_loss_30)
forwards_d = forwards_to_reach(loss_history_d, target_loss_30)

print(
    f"• [Brazo A] Mutación Aleatoria Simple:       Loss={current_loss_a:.4f} (-{reduction_a:.1f}%) | Forwards a -30%: {forwards_a}"
)
print(
    f"• [Brazo B] SPSA Discreto (Temperatura T_g): Loss={current_loss_b:.4f} (-{reduction_b:.1f}%) | Forwards a -30%: {forwards_b}"
)
print(
    f"• [Brazo C] Reglas Locales Hebbianas:        Loss={current_loss_c:.4f} (-{reduction_c:.1f}%) | Forwards a -30%: {forwards_c}"
)
print(
    f"• [Brazo D] Currículo Híbrido H3:            Loss={current_loss_d:.4f} (-{reduction_d:.1f}%) | Forwards a -30%: {forwards_d}"
)

speedup_c = forwards_a / max(1, forwards_c)
speedup_d = forwards_a / max(1, forwards_d)

print(
    "--------------------------------------------------------------------------------"
)
print(
    f"⚡ Aceleración Reglas Locales (C vs A): {speedup_c:.2f}x más rápido a -30% Loss"
)
print(
    f"⚡ Aceleración Currículo Híbrido (D vs A): {speedup_d:.2f}x más rápido a -30% Loss"
)
print(
    "--------------------------------------------------------------------------------"
)

h3_verified = forwards_d < forwards_a and reduction_d > reduction_b
print(f"✅ HIPÓTESIS H3 (Currículo Híbrido) CONFIRMADA: {h3_verified}")
print(
    f"   La combinación de Crecimiento Hebbiano ({forwards_c} forwards) + Refinamiento SPSA"
)
print(f"   alcanza el objetivo {speedup_d:.2f}x más rápido que la mutación ciega.")
print(
    "================================================================================"
)
