#!/usr/bin/env python3
# diagnose_phase_capacity.py
# Diagnóstico empírico de fase vs capacidad para modelos GAJE cuantizados.
# Cinco ajustes integrados:
#   1) SVD con N >= D (o advertencia explícita)
#   2) Test estadístico z-score para el baseline nulo
#   3) ER centrado y ER crudo por separado
#   4) Tests sintéticos independientes (solo-fase y solo-magnitud)
#   5) Fracciones alineada/rotada/fragmentada del subespacio top-k

import sys
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ---------------------------------------------------------------------------
# Utilidades base
# ---------------------------------------------------------------------------

def decompose_polar(x: np.ndarray):
    """
    x shape: [N, L, D]
    Retorna:
      norms: [N, L, 1]
      u_hat: [N, L, D]  (vectores unitarios)
    """
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms_safe = np.where(norms == 0, 1e-12, norms)
    u_hat = x / norms_safe
    return norms, u_hat


def effective_rank(singular_values: np.ndarray) -> float:
    """
    Participation ratio / exponencial de la entropía de Shannon de los
    valores singulares:
        p_k = s_k / sum(s)
        ER  = exp( -sum(p_k * ln(p_k)) )
    Mide cuántas dimensiones ortogonales albergan realmente la varianza.
    """
    s = singular_values[singular_values > 1e-12]
    if len(s) == 0:
        return 0.0
    p = s / np.sum(s)
    return float(np.exp(-np.sum(p * np.log(p))))


def warn_if_flaco(N: int, D: int):
    """Advierte si N < D, porque el SVD devolverá a lo sumo N singulares."""
    if N < D:
        print(f"[!] ADVERTENCIA: N={N} < D={D}. El SVD devolverá como máximo "
              f"{N} valores singulares y el ER estará truncado artificialmente. "
              f"Usa al menos {D} tokens (idealmente 2-3×D) para una medida fiable.")


# ---------------------------------------------------------------------------
# Métricas espectrales y de subespacio (ajustes 3 y 5)
# ---------------------------------------------------------------------------

def spectral_metrics(X: np.ndarray, k_top: int):
    """
    X: [N, D] activaciones en una capa (sin centrar).
    Devuelve un dict con:
        er_raw      : rango efectivo sobre datos crudos
        er_centered : rango efectivo sobre datos centrados
        cos_top     : cosenos principales entre control y Q2 (más tarde)
    """
    # ER sobre datos crudos
    _, s_raw, _ = np.linalg.svd(X, full_matrices=False)
    er_raw = effective_rank(s_raw)

    # ER sobre datos centrados (elimina la componente DC / modo común)
    Xc = X - np.mean(X, axis=0, keepdims=True)
    _, s_c, Vt_c = np.linalg.svd(Xc, full_matrices=False)
    er_centered = effective_rank(s_c)

    return {
        "er_raw": er_raw,
        "er_centered": er_centered,
        "s_c": s_c,
        "Vt_c_top": Vt_c[:k_top, :],
    }


def subspace_fractions(Vt_ctrl_top: np.ndarray, Vt_q2_top: np.ndarray):
    """
    Fracción de direcciones principales alineadas / rotadas / fragmentadas
    entre los top-k subespacios del control y de Q2.
    """
    M = Vt_ctrl_top @ Vt_q2_top.T
    cos_principal = np.linalg.svd(M, compute_uv=False)
    frac_aligned = float(np.mean(cos_principal > 0.9))
    frac_rotated = float(np.mean(cos_principal < 0.5))
    frac_mid = 1.0 - frac_aligned - frac_rotated
    return {
        "cos_principal": cos_principal,
        "frac_aligned": frac_aligned,
        "frac_rotated": frac_rotated,
        "frac_mid": frac_mid,
    }


# ---------------------------------------------------------------------------
# Diagnóstico principal
# ---------------------------------------------------------------------------

def run_diagnostics(x_ctrl: np.ndarray, x_q2: np.ndarray,
                    z_thresholds=(1.5, 3.0),
                    output_plot="diagnostico_fase_capacidad.png"):
    """
    x_ctrl, x_q2: [N, L, D] activaciones por token, capa y dimensión.
    z_thresholds: cortes para interpretar z-score del coseno contra el null.
    """
    assert x_ctrl.shape == x_q2.shape, \
        f"Dim mismatch: {x_ctrl.shape} vs {x_q2.shape}"
    N, L, D = x_ctrl.shape
    print(f"[*] Procesando {N} tokens, {L} capas, D={D}")
    warn_if_flaco(N, D)

    # ------------------------------------------------------------------
    # Ajuste 1: descomposición polar de señal y de baseline nulo
    # ------------------------------------------------------------------
    r_ctrl, u_ctrl = decompose_polar(x_ctrl)
    r_q2, u_q2 = decompose_polar(x_q2)

    raw_random_dirs = np.random.randn(N, L, D).astype(np.float32)
    _, u_random = decompose_polar(raw_random_dirs)

    # ------------------------------------------------------------------
    # Ajuste 2: estadística del coseno + z-score
    # ------------------------------------------------------------------
    sc_real = np.sum(u_ctrl * u_q2, axis=-1)      # [N, L]
    sc_null = np.sum(u_ctrl * u_random, axis=-1)  # [N, L]

    mean_sc_real = np.mean(sc_real, axis=0)
    std_sc_real = np.std(sc_real, axis=0)
    mean_sc_null = np.mean(sc_null, axis=0)
    std_sc_null = np.std(sc_null, axis=0)

    # Error estándar de la media del null, por capa
    se_null = std_sc_null / np.sqrt(N)
    se_null = np.where(se_null < 1e-12, 1e-12, se_null)
    z_score = (mean_sc_real - mean_sc_null) / se_null

    # ------------------------------------------------------------------
    # Ajuste 3: ER crudo y centrado, y top-k subespacios
    # ------------------------------------------------------------------
    k_top = min(8, D, N)
    er_raw_ctrl = np.zeros(L)
    er_raw_q2 = np.zeros(L)
    er_cen_ctrl = np.zeros(L)
    er_cen_q2 = np.zeros(L)
    frac_aligned = np.zeros(L)
    frac_rotated = np.zeros(L)
    frac_mid = np.zeros(L)

    for l in range(L):
        m_ctrl = spectral_metrics(x_ctrl[:, l, :], k_top)
        m_q2 = spectral_metrics(x_q2[:, l, :], k_top)

        er_raw_ctrl[l] = m_ctrl["er_raw"]
        er_raw_q2[l] = m_q2["er_raw"]
        er_cen_ctrl[l] = m_ctrl["er_centered"]
        er_cen_q2[l] = m_q2["er_centered"]

        sub = subspace_fractions(m_ctrl["Vt_c_top"], m_q2["Vt_c_top"])
        frac_aligned[l] = sub["frac_aligned"]
        frac_rotated[l] = sub["frac_rotated"]
        frac_mid[l] = sub["frac_mid"]

    # ------------------------------------------------------------------
    # Reporte en terminal
    # ------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("CAPA | S_C(Q2,Ctrl)      | z-score | Ratio||x||(Q2/Ctrl) | ER_raw C/Q | ER_cen C/Q | alin/rot/frac")
    print("=" * 110)
    mean_ratio_real = np.mean((r_q2 / (r_ctrl + 1e-12)).squeeze(-1), axis=0)
    std_ratio_real = np.std((r_q2 / (r_ctrl + 1e-12)).squeeze(-1), axis=0)
    for l in range(L):
        print(f" {l:2d}  | {mean_sc_real[l]:.3f} ± {std_sc_real[l]:.3f} | "
              f"{z_score[l]:6.2f}  | "
              f"{mean_ratio_real[l]:.3f} ± {std_ratio_real[l]:.3f}     | "
              f"{er_raw_ctrl[l]:5.1f}/{er_raw_q2[l]:5.1f} | "
              f"{er_cen_ctrl[l]:5.1f}/{er_cen_q2[l]:5.1f} | "
              f"{frac_aligned[l]:.2f}/{frac_rotated[l]:.2f}/{frac_mid[l]:.2f}")
    print("=" * 110)

    # Veredictos automáticos por capa
    print("\n[*] Veredicto automático por capa (z-score vs null):")
    for l in range(L):
        z = z_score[l]
        if z < z_thresholds[0]:
            v = "colapso angular total"
        elif z < z_thresholds[1]:
            v = "preservación marginal"
        else:
            v = "preservación significativa"
        print(f"    capa {l:2d}: z={z:5.2f} -> {v}")

    # ------------------------------------------------------------------
    # Gráficas
    # ------------------------------------------------------------------
    if HAS_MATPLOTLIB:
        fig, axs = plt.subplots(1, 3, figsize=(19, 5))
        layers = np.arange(L)

        # Panel 1: coseno vs null con banda de ±1σ
        axs[0].plot(layers, mean_sc_real, marker='o', color='tab:blue',
                    label="S_C(Q2, Ctrl)")
        axs[0].fill_between(layers,
                            mean_sc_real - std_sc_real,
                            mean_sc_real + std_sc_real,
                            color='tab:blue', alpha=0.15)
        axs[0].plot(layers, mean_sc_null, linestyle='--', color='gray',
                    label="Null (fase aleatoria)")
        axs[0].axhline(0.0, color='r', linestyle=':', alpha=0.5)
        axs[0].set_title("Preservación angular vs ruido nulo")
        axs[0].set_xlabel("Capa")
        axs[0].set_ylabel("Similitud coseno")
        axs[0].set_ylim(-0.2, 1.05)
        axs[0].grid(True, alpha=0.3)
        axs[0].legend()

        # Panel 2: magnitud y ER
        ax2 = axs[1]
        ax2.plot(layers, mean_ratio_real, marker='s', color='tab:orange',
                 label="||x_Q2|| / ||x_ctrl||")
        ax2.fill_between(layers,
                         mean_ratio_real - std_ratio_real,
                         mean_ratio_real + std_ratio_real,
                         color='tab:orange', alpha=0.15)
        ax2.axhline(1.0, color='g', linestyle='--', alpha=0.5,
                    label="Preservación exacta")
        ax2.set_title("Evolución radial")
        ax2.set_xlabel("Capa")
        ax2.set_ylabel("Escala relativa")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Panel 3: ER crudo y centrado + fracciones
        ax3 = axs[2]
        ax3.plot(layers, er_raw_ctrl, marker='^', color='tab:green',
                 label="ER raw Ctrl")
        ax3.plot(layers, er_raw_q2, marker='v', color='tab:red', linestyle='--',
                 label="ER raw Q2")
        ax3.plot(layers, er_cen_ctrl, marker='^', color='tab:green', alpha=0.5,
                 linestyle=':', label="ER centered Ctrl")
        ax3.plot(layers, er_cen_q2, marker='v', color='tab:red', linestyle=':',
                 alpha=0.5, label="ER centered Q2")
        ax3.set_title("Rango efectivo (raw y centrado)")
        ax3.set_xlabel("Capa")
        ax3.set_ylabel("Dimensiones activas")
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(output_plot, dpi=300)
        print(f"\n[✓] Gráficas guardadas en '{output_plot}'")
    else:
        print("\n[i] matplotlib no disponible; reporte estadístico impreso en terminal.")

    return {
        "mean_sc_real": mean_sc_real,
        "std_sc_real": std_sc_real,
        "mean_sc_null": mean_sc_null,
        "z_score": z_score,
        "mean_ratio_real": mean_ratio_real,
        "er_raw_ctrl": er_raw_ctrl,
        "er_raw_q2": er_raw_q2,
        "er_cen_ctrl": er_cen_ctrl,
        "er_cen_q2": er_cen_q2,
        "frac_aligned": frac_aligned,
        "frac_rotated": frac_rotated,
        "frac_mid": frac_mid,
    }


# ---------------------------------------------------------------------------
# Ajuste 4: tests sintéticos independientes (fase vs magnitud)
# ---------------------------------------------------------------------------

def _test_phase_only(ctrl, L_from=3):
    """Rota las capas >= L_from con una Q ortogonal. Magnitud intacta."""
    q2 = ctrl.copy()
    D = ctrl.shape[-1]
    for l in range(L_from, ctrl.shape[1]):
        rot = np.random.randn(D, D).astype(np.float32)
        q, _ = np.linalg.qr(rot)
        q2[:, l, :] = q2[:, l, :] @ q.T  # rotación unitaria
    return q2


def _test_magnitude_only(ctrl, L_from=3, factor=0.5):
    """Escala las capas >= L_from. Fase intacta."""
    q2 = ctrl.copy()
    q2[:, L_from:, :] *= factor
    return q2


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Diagnóstico de Fase vs Magnitud para modelos GAJE")
    parser.add_argument("--data", type=str, default=None, help="Archivo .npz con tensores 'ctrl' y 'q2' [N, L, D]")
    parser.add_argument("--synthetic", action="store_true", default=False, help="Forzar ejecución de tests sintéticos A y B")
    args = parser.parse_args()

    # Semilla para reproducibilidad
    np.random.seed(42)

    if args.data is not None:
        data = np.load(args.data)
        x_ctrl = data["ctrl"]
        x_q2 = data["q2"]
        print(f"\n[+] Cargados datos reales desde {args.data}")
        run_diagnostics(x_ctrl, x_q2, output_plot="diagnostico_real_fase_capacidad.png")
    else:
        # Por defecto corre los tests sintéticos A y B para calibración y sanity check
        N_t, L_t, D_t = 64, 8, 256
        ctrl = np.random.randn(N_t, L_t, D_t).astype(np.float32)

        # ---------------------------------------------------------------
        # Test A: solo fase (magnitud intacta)
        # Esperado: S_C cae en capas 3-7, ratio ≈ 1, ER cambia por rotación.
        # ---------------------------------------------------------------
        print("\n" + "#" * 80)
        print("# TEST A: solo fase (rotación unitaria, magnitud intacta)")
        print("#" * 80)
        q2_phase = _test_phase_only(ctrl, L_from=3)
        run_diagnostics(ctrl, q2_phase, output_plot="diagnostico_sintetico_test_A.png")

        # ---------------------------------------------------------------
        # Test B: solo magnitud (fase intacta)
        # Esperado: S_C ≈ 1, ratio ≈ 0.5, ER baja por escala.
        # ---------------------------------------------------------------
        print("\n" + "#" * 80)
        print("# TEST B: solo magnitud (escala 0.5, dirección intacta)")
        print("#" * 80)
        q2_mag = _test_magnitude_only(ctrl, L_from=3, factor=0.5)
        run_diagnostics(ctrl, q2_mag, output_plot="diagnostico_sintetico_test_B.png")
