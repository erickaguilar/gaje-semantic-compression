import numpy as np
from gaje.utils.metrics import calculate_activation_drift


class SignalToNoiseBalancer:
    """
    Controlador dinámico que ajusta el metabolismo genómico (anclas y precisión)
    para mantener la homeostasis de la señal semántica.
    """

    def __init__(self, target_drift=0.0015, min_threshold=0.02, max_threshold=0.4):
        self.target_drift = target_drift
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.current_threshold = 0.15
        self.drift_history = []
        self.threshold_history = []

    def adjust(self, act_orig, act_gen):
        """
        Ajusta el umbral de anclas basado en el drift de activación detectado.
        """
        drift = calculate_activation_drift(act_orig, act_gen)
        self.drift_history.append(drift)

        # PID simple o lógica proporcional para el umbral
        if drift > self.target_drift:
            # Demasiado ruido -> bajar umbral para capturar más anclas
            adjustment = 0.90
        else:
            # Señal limpia -> subir umbral para ahorrar memoria/computo
            adjustment = 1.10

        self.current_threshold *= adjustment
        self.current_threshold = np.clip(
            self.current_threshold, self.min_threshold, self.max_threshold
        )
        self.threshold_history.append(self.current_threshold)

        return self.current_threshold

    def generate_precision_mask(self, entropy_per_dim, fidelity_level=0.8):
        """
        Genera una máscara de precisión (Fase 12) basada en la entropía de Shannon.
        """
        try:
            from gaje.core import _impl as dna_core

            mask_bytes = dna_core.generate_precision_mask_native(
                entropy_per_dim.tolist(), fidelity_level
            )
            return np.frombuffer(mask_bytes, dtype=np.uint8).copy()
        except (ImportError, AttributeError):
            q_mid = np.quantile(entropy_per_dim, 1.0 - fidelity_level)
            q_high = np.quantile(entropy_per_dim, 1.0 - (fidelity_level / 2))

            mask = np.zeros_like(entropy_per_dim, dtype=np.uint8)
            mask[entropy_per_dim > q_mid] = 1
            mask[entropy_per_dim > q_high] = 2
            return mask

    def prune_dimensions(self, database, stride, entropy_per_dim, threshold=0.01):
        """
        Elimina dimensiones redundantes (Neural Pruning DNA - Fase 12).
        """
        from gaje.core import _impl as dna_core

        try:
            active_dims = dna_core.get_active_dimensions_native(
                entropy_per_dim.tolist(), threshold
            )
        except AttributeError:
            active_dims = np.where(entropy_per_dim > threshold)[0].tolist()

        if len(active_dims) == len(entropy_per_dim):
            return database, active_dims

        print(
            f"✂️ Neural Pruning: Eliminando {len(entropy_per_dim) - len(active_dims)} dimensiones redundantes."
        )
        new_db, new_stride = dna_core.prune_genomic_database(
            database, stride, active_dims
        )

        return new_db, active_dims

    def report(self):
        if not self.drift_history:
            return "No data collected."

        avg_drift = np.mean(self.drift_history)
        last_threshold = self.current_threshold
        return f"[SN Balancer] Avg Drift: {avg_drift:.6f} | Current Threshold: {last_threshold:.4f}"
