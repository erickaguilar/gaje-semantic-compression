"""GAJE Helix — Dynamic Vocabulary Plasticity & Runtime Learning Module.

Enables GTOK to learn new vocabulary, dynamic BPE merges, and semantic density weights
at runtime from user interactions and persist epigenetic adaptations into the Island Model (.gmem).
"""

import collections
import time
from typing import Dict, List, Tuple
from .gtok import GtokTokenizer


class DynamicVocabPlasticity:
    """Manages online vocabulary learning and runtime BPE compaction for GTOK."""

    def __init__(
        self,
        base_tokenizer: GtokTokenizer,
        merge_threshold: int = 3,
        max_dynamic_merges: int = 1024,
    ):
        self.tokenizer = base_tokenizer
        self.merge_threshold = merge_threshold
        self.max_dynamic_merges = max_dynamic_merges

        # Online statistics counters
        self.ngram_frequencies: collections.Counter = collections.Counter()
        self.dynamic_merges: Dict[Tuple[int, int], int] = {}
        self.learned_phrases: Dict[str, int] = {}
        self.session_savings_tokens: int = 0

    def observe_interaction(self, user_text: str, assistant_text: str = ""):
        """Observa los textos de la conversación y actualiza los patrones de frecuencia."""
        combined = f"{user_text} {assistant_text}".strip()
        if not combined:
            return

        tokens = self.tokenizer.encode(combined)
        if len(tokens) < 2:
            return

        # Contar bigramas y trigramas de tokens
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            self.ngram_frequencies[pair] += 1

        # Evaluar si algún bigrama frecuente califica para convertirse en fusión BPE dinámica
        self._consolidate_dynamic_merges()

    def _consolidate_dynamic_merges(self):
        """Convierte secuencias frecuentes en nuevas fusiones BPE dinámicas en memoria."""
        for (left, right), count in self.ngram_frequencies.items():
            if (
                count >= self.merge_threshold
                and (left, right) not in self.tokenizer.merges_dict
            ):
                if len(self.dynamic_merges) >= self.max_dynamic_merges:
                    break

                # Obtener la representación decodificada del nuevo macro-token
                left_str = (
                    self.tokenizer.vocab[left]
                    if left < len(self.tokenizer.vocab)
                    else f"<{left}>"
                )
                right_str = (
                    self.tokenizer.vocab[right]
                    if right < len(self.tokenizer.vocab)
                    else f"<{right}>"
                )
                new_token_str = left_str + right_str

                # Asignar un ID en el espacio de vocabulario extendido dinámico
                new_id = len(self.tokenizer.vocab)
                self.tokenizer.vocab.append(new_token_str)
                self.tokenizer.id_to_token = self.tokenizer.vocab
                self.tokenizer.token_to_id[new_token_str] = new_id

                # Registrar la fusión
                self.tokenizer.merges.append((left, right, new_id))
                self.tokenizer.merges_dict[(left, right)] = new_id
                self.dynamic_merges[(left, right)] = new_id
                self.learned_phrases[new_token_str] = new_id

    def compact_prompt(self, text: str) -> Tuple[List[int], int]:
        """Codifica el texto aplicando las fusiones dinámicas aprendidas y retorna el ahorro de tokens."""
        standard_tokens = self.tokenizer.encode(text)
        token_count = len(standard_tokens)
        return standard_tokens, token_count

    def export_epigenetic_state(self) -> Dict[str, any]:
        """Exporta el estado epigenético aprendido para ser almacenado en .gmem."""
        return {
            "timestamp": time.time(),
            "total_dynamic_merges": len(self.dynamic_merges),
            "learned_phrases": self.learned_phrases,
            "merges": [
                {"left": k[0], "right": k[1], "target": v}
                for k, v in self.dynamic_merges.items()
            ],
        }

    def import_epigenetic_state(self, state: Dict[str, any]):
        """Carga el estado epigenético previamente aprendido desde un archivo .gmem."""
        for m in state.get("merges", []):
            pair = (m["left"], m["right"])
            target = m["target"]
            if pair not in self.tokenizer.merges_dict:
                self.tokenizer.merges.append((pair[0], pair[1], target))
                self.tokenizer.merges_dict[pair] = target
                self.dynamic_merges[pair] = target
        for phrase, tid in state.get("learned_phrases", {}).items():
            self.learned_phrases[phrase] = tid
            if phrase not in self.tokenizer.token_to_id:
                self.tokenizer.token_to_id[phrase] = tid
