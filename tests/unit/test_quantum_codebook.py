"""Unit tests for GAJE Quantum Superposition Meta-Tokens Codebook (Phase 1 & 2)."""

import os
import tempfile
import unittest
import numpy as np
from python.gaje.processing.quantum_codebook import (
    QuantumCodebook,
    QuantumEmbeddingTable,
)


class TestQuantumCodebook(unittest.TestCase):
    def test_harmonic_codebook_normalization(self):
        K = 256
        dim = 128
        codebook = QuantumCodebook.create_harmonic_codebook(num_meta_tokens=K, dim=dim)
        self.assertEqual(codebook.centroids.shape, (K, dim))

        # Verificar que todos los centroides estén normalizados a norma 1.0
        norms = np.linalg.norm(codebook.centroids, axis=1)
        for n in norms:
            self.assertAlmostEqual(float(n), 1.0, places=5)

    def test_sparse_superposition_projection_and_reconstruction(self):
        K = 512
        dim = 64
        num_tokens = 2000
        m = 4

        # Crear embeddings sintéticos de prueba
        np.random.seed(42)
        dense_embeddings = np.random.randn(num_tokens, dim).astype(np.float32)

        codebook = QuantumCodebook(num_meta_tokens=K, dim=dim)
        codebook.fit_from_embeddings(
            dense_embeddings, num_iterations=5, batch_size=1000
        )

        # Proyectar y reconstruir un vector
        vec = dense_embeddings[0]
        inds, amps = codebook.project_sparse(vec, m=m)
        self.assertEqual(len(inds), m)
        self.assertEqual(len(amps), m)

        # Verificar que la norma de las amplitudes sea 1.0
        amp_norm = np.linalg.norm(amps)
        self.assertAlmostEqual(float(amp_norm), 1.0, places=5)

        reconstructed = codebook.reconstruct(inds, amps)
        self.assertEqual(len(reconstructed), dim)

        # Similitud coseno entre original y reconstruido
        cos_sim = float(np.dot(vec / np.linalg.norm(vec), reconstructed))
        self.assertGreater(cos_sim, 0.70)

    def test_qemb_table_binary_roundtrip_and_compression(self):
        K = 128
        dim = 32
        num_tokens = 500
        m = 4

        dense = np.random.randn(num_tokens, dim).astype(np.float32)
        table = QuantumEmbeddingTable.from_dense_embeddings(
            dense, num_meta_tokens=K, m=m
        )

        with tempfile.NamedTemporaryFile(suffix=".qemb", delete=False) as tf:
            qemb_path = tf.name

        try:
            table.save_qemb(qemb_path)
            self.assertTrue(os.path.exists(qemb_path))

            # Verificar tamaño binario: K*dim*4 + V*m*2 + V*m*1 + 64 bytes
            file_size = os.path.getsize(qemb_path)
            raw_dense_size = num_tokens * dim * 4
            print(
                f"\n[QEMB TEST] Tamaño denso FP32: {raw_dense_size} B | Comprimido .qemb: {file_size} B"
            )

            # Recargar
            reloaded = QuantumEmbeddingTable.load_qemb(qemb_path)
            self.assertEqual(reloaded.num_tokens, num_tokens)
            self.assertEqual(reloaded.m, m)
            self.assertEqual(reloaded.codebook.num_meta_tokens, K)

            # Verificar lookup
            emb_orig = table.get_embedding(10)
            emb_reloaded = reloaded.get_embedding(10)
            np.testing.assert_allclose(emb_orig, emb_reloaded, atol=1e-4)
        finally:
            if os.path.exists(qemb_path):
                os.remove(qemb_path)


if __name__ == "__main__":
    unittest.main()
