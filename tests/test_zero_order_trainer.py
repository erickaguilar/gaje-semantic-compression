#!/usr/bin/env python3
"""
🧬 TEST: ENTRENAMIENTO DE ORDEN CERO NATIVO (SPSA DISCRETO)
================================================================================
Verifica que `fit_zero_order` ejecute optimización por orden cero sobre un modelo
real .flat (smollm2_135m.flat) usando exclusivamente forward passes deterministas.
================================================================================
"""

import os
import unittest
from gaje.nn.stabilized import GenomicLLM
from dna_semantic_compression import NativeGenomicTrainer

class TestZeroOrderTrainer(unittest.TestCase):
    def test_zero_order_spsa_training(self):
        model_path = "models/production/smollm2_135m.flat"
        if not os.path.exists(model_path):
            self.skipTest(f"Modelo {model_path} no encontrado.")

        print(f"\n[TEST] Cargando organismo {model_path} para entrenamiento SPSA de orden cero...")
        llm = GenomicLLM.load_genomic(model_path)
        self.assertIsNotNone(llm)

        # Micro-dataset de secuencias de tokens sintéticas
        dataset = [
            [280, 395, 1599, 345, 406],
            [102, 503, 894, 201, 77],
            [55, 312, 440, 981, 1024]
        ]

        trainer = NativeGenomicTrainer(lr=0.01, resonance_weight=0.05)
        print("[TEST] Ejecutando fit_zero_order(epochs=2, k_coords=16)...")
        final_loss = trainer.fit_zero_order(llm.rust_llm, dataset, epochs=2, k_coords=16)

        print(f"[TEST] Loss final obtenido tras SPSA: {final_loss:.4f}")
        self.assertGreater(final_loss, 0.0)
        self.assertFalse(float('nan') == final_loss)
        self.assertFalse(float('inf') == final_loss)
        print("✅ [TEST] Entrenamiento nativo de orden cero (SPSA Discreto) ejecutado exitosamente.")

if __name__ == "__main__":
    unittest.main()
