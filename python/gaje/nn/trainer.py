import numpy as np
import time

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


class GenomicTrainer:
    def __init__(self, model, lr=0.01, use_torch=True, resonance_weight=0.05):
        self.model = model
        self.lr = lr
        self.resonance_weight = resonance_weight
        self.use_torch = use_torch and HAS_TORCH
        if self.use_torch:
            print(
                f"[*] GenomicTrainer: Usando PyTorch (Resonance Weight: {resonance_weight})"
            )
        else:
            print(
                f"[*] GenomicTrainer: Usando NumPy (Resonance Weight: {resonance_weight})"
            )

    def _tokenize(self, text):
        if not hasattr(self.model, "tokenizer") or self.model.tokenizer is None:
            return None

        try:
            encoding = self.model.tokenizer.encode(text)
            if hasattr(encoding, "ids"):
                return encoding.ids
            if hasattr(encoding, "get_ids"):
                return encoding.get_ids()
            return encoding
        except Exception:
            try:
                return self.model.tokenizer.encode(text, add_special_tokens=False)
            except Exception:
                return None

    def train_step(self, input_ids, target_ids, phase=1):
        """
        Executes a single training step with Semantic Resonance Loss and Phase-aware refinement.
        """
        seq_len = len(input_ids)
        self.model.rust_llm.clear_cache_py()
        loss_total = 0.0

        # Para fases avanzadas (2 y 3), usamos el optimizador nativo de secuencias en Rust
        # para refinamiento profundo, pero el LM Head se refina aquí con Resonance Loss.
        if phase >= 2:
            # Entrenamiento profundo en Rust (Refina bloques y LM Head con Cross-Entropy estándar)
            # Esto proporciona la base de estabilidad.
            self.model.rust_llm.train_on_sequence(
                input_ids + [target_ids[-1]], self.lr * 0.5
            )

        for i in range(seq_len):
            token_id = int(input_ids[i])
            target_id = int(target_ids[i])

            logits_raw, h_norm_raw = self.model.rust_llm.forward_with_hidden(
                token_id, False
            )

            if self.use_torch:
                logits = torch.tensor(logits_raw, requires_grad=True)
                target = torch.tensor([target_id], dtype=torch.long)

                # 1. Cross Entropy Loss
                ce_loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), target)

                # 2. Semantic Resonance Loss (Entropy Penalty)
                # Castiga distribuciones planas, forzando al modelo a ser decisivo.
                probs = torch.nn.functional.softmax(logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-12))

                loss = ce_loss + self.resonance_weight * entropy
                loss_total += loss.item()

                loss.backward()
                grad_logits = logits.grad.numpy()
            else:
                probs = softmax(np.array(logits_raw))
                ce_loss = -np.log(probs[target_id] + 1e-12)

                # Gradiente manual de Entropía: p * (-ln p - H)
                entropy = -np.sum(probs * np.log(probs + 1e-12))
                loss = ce_loss + self.resonance_weight * entropy
                loss_total += loss

                grad_logits = probs.copy()
                grad_logits[target_id] -= 1.0
                grad_entropy = probs * (-np.log(probs + 1e-12) - entropy)
                grad_logits += self.resonance_weight * grad_entropy

            # Siempre refinamos el LM Head con el gradiente que incluye Resonancia
            self.model.rust_llm.refine_lm_head(
                h_norm_raw, grad_logits.tolist(), self.lr
            )

            # En Fase 3, activamos mutaciones homeostáticas leves
            if phase >= 3 and i % 8 == 0:
                self.model.rust_llm.mutate_all_homeostasis(self.lr * 0.01)

        return loss_total / seq_len

    def fit(self, dataset, epochs=10):
        print(f"[*] Iniciando entrenamiento Born-Genomic ({epochs} épocas)")

        # Definición de fases del Curriculum Learning
        p1_end = int(epochs * 0.2)  # 20% Base (LM Head)
        p2_end = int(epochs * 0.7)  # 50% IQAT (Bloques + LM Head)

        for epoch in range(epochs):
            # Determinar fase actual
            if epoch < p1_end:
                phase = 1
                phase_name = "Base (LM Head Only)"
            elif epoch < p2_end:
                phase = 2
                phase_name = "IQAT (Deep Refinement)"
            else:
                phase = 3
                phase_name = "Evol (Homeostatic Mutation)"

            total_loss = 0
            count = 0
            start_time = time.time()

            for text in dataset:
                tokens = self._tokenize(text)
                if tokens is None or len(tokens) < 2:
                    continue

                loss = self.train_step(tokens[:-1], tokens[1:], phase=phase)
                total_loss += loss
                count += 1

            duration = time.time() - start_time
            if count > 0:
                avg_loss = total_loss / count
                ppl = np.exp(avg_loss) if avg_loss < 20 else 999.9
                print(
                    f"    - Época {epoch + 1}/{epochs} [{phase_name}] | Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | {duration:.2f}s"
                )
            else:
                print(f"    - Época {epoch + 1}/{epochs} | Sin datos válidos")

    def evaluate(self, dataset):
        total_loss = 0
        count = 0
        for text in dataset:
            tokens = self._tokenize(text)
            if tokens is None or len(tokens) < 2:
                continue

            loss = self.model.rust_llm.train_on_sequence(tokens, 0.0)
            total_loss += loss
            count += 1
        return total_loss / count if count > 0 else 0
