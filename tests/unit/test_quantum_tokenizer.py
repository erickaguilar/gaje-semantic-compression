"""Unit tests for GAJE Quantum Genomic Tokenizer prototype."""

import unittest
import numpy as np
from python.gaje.processing.quantum_tokenizer import QuantumGenomicTokenizer


class TestQuantumGenomicTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = QuantumGenomicTokenizer()

    def test_density_matrix_trace_and_purity(self):
        state = self.tokenizer.encode_char_to_state("A")
        self.assertAlmostEqual(state.trace, 1.0, places=5)
        self.assertAlmostEqual(state.purity, 1.0, delta=0.01)
        self.assertGreaterEqual(state.von_neumann_entropy, 0.0)

    def test_born_rule_contextual_collapse(self):
        state = self.tokenizer.encode_char_to_state("X")
        # Contexto sesgado fuertemente hacia Guanina
        ctx_g = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.complex128)
        base, conf = state.collapse_with_context(ctx_g)
        self.assertEqual(base, "G")
        self.assertGreater(conf, 0.5)

    def test_text_to_dna_encoding(self):
        text = "GAJE Helix Engine"
        dna = self.tokenizer.collapse_text_to_dna(
            text, context_text="Biología Molecular y Cuántica"
        )
        self.assertEqual(len(dna), len(text))
        for nucleotide in dna:
            self.assertIn(nucleotide, ["A", "C", "G", "T"])


if __name__ == "__main__":
    unittest.main()
