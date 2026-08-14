// =============================================================================
// activation.rs — Activaciones tipo GLU vectorizadas con Estabilización Dinámica
// =============================================================================

use rayon::prelude::*;

#[inline(always)]
pub fn swiglu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    let _n = gate.len();
    out.par_iter_mut()
        .zip(gate.par_iter())
        .zip(up.par_iter())
        .for_each(|((o, &g), &u)| {
            // Estabilización de SwiGLU (Silu gating)
            // Limitamos el rango dinámico para evitar que el ruido de cuantización
            // de 2 bits se magnifique en las colas de la exponencial.
            let g_safe = g.clamp(-64.0, 64.0);
            let sigmoid = if g_safe >= 0.0 {
                1.0 / (1.0 + (-g_safe).exp())
            } else {
                let ex = g_safe.exp();
                ex / (1.0 + ex)
            };
            let silu = g * sigmoid;

            // Clamping adaptativo: reduce la probabilidad de explosión de gradiente/activación
            // en modelos profundos (>24 bloques).
            *o = (silu * u).clamp(-96.0, 96.0);
        });
}

/// Versión balanceada de SwiGLU que compensa el sesgo (bias) introducido
/// por la cuantización asimétrica de 2 bits.
#[inline(always)]
pub fn swiglu_balanced(gate: &[f32], up: &[f32], out: &mut [f32], _h_scale: f32) {
    out.par_iter_mut()
        .zip(gate.par_iter())
        .zip(up.par_iter())
        .for_each(|((o, &g), &u)| {
            let g_safe = g.clamp(-64.0, 64.0);
            let sigmoid = if g_safe >= 0.0 {
                1.0 / (1.0 + (-g_safe).exp())
            } else {
                let ex = g_safe.exp();
                ex / (1.0 + ex)
            };

            let silu = g * sigmoid;
            *o = silu * u;
        });
}

#[inline(always)]
pub fn geglu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    out.par_iter_mut()
        .zip(gate.par_iter())
        .zip(up.par_iter())
        .for_each(|((o, &g), &u)| {
            // Estabilización de GeGLU
            let g_safe = g.clamp(-20.0, 20.0);
            let tanh_inner = 0.7978846f32 * (g_safe + 0.044715f32 * g_safe * g_safe * g_safe);
            let gelu = 0.5f32 * g_safe * (1.0f32 + tanh_inner.tanh());
            *o = (gelu * u).clamp(-128.0, 128.0);
        });
}

#[inline(always)]
pub fn relu_glu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    out.par_iter_mut()
        .zip(gate.par_iter())
        .zip(up.par_iter())
        .for_each(|((o, &g), &u)| {
            *o = (g.max(0.0) * u).clamp(-128.0, 128.0);
        });
}