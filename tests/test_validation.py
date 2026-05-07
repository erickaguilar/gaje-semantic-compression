import unittest
import numpy as np
import dna_semantic_compression

class TestDNASemantic(unittest.TestCase):
    def test_quantize_dequantize_dims(self):
        """Test if dimensions are preserved after a round trip."""
        dims = 768
        original_vector = np.random.uniform(-1, 1, dims).astype(np.float32).tolist()
        
        # Quantize
        dna_packed = dna_semantic_compression.quantize_embedding(original_vector)
        # 768 dims / 4 dims per byte = 192 bytes
        self.assertEqual(len(dna_packed), 192)
        
        # Dequantize
        reconstructed = dna_semantic_compression.dequantize_embedding(dna_packed, dims)
        self.assertEqual(len(reconstructed), dims)

    def test_quantization_logic(self):
        """Test Gray Code quantization levels."""
        # Thresholds: -0.34, 0.0, 0.34
        # A < -0.34 (00)
        # C < 0.0   (01)
        # G < 0.34  (11) - Gray Code
        # T >= 0.34 (10) - Gray Code
        vector = [-0.5, -0.1, 0.1, 0.5] # A, C, G, T
        dna_packed = dna_semantic_compression.quantize_embedding(vector)
        
        # Binary: 00 01 11 10 = 0b00011110 (30 decimal)
        self.assertEqual(dna_packed[0], 0b00011110)

    def test_search_adc_consistency(self):
        """Test if ADC search finds the exact match with low distance."""
        dims = 128
        db_vectors = [np.random.uniform(-1, 1, dims).astype(np.float32).tolist() for _ in range(10)]
        db_dna = [dna_semantic_compression.quantize_embedding(v) for v in db_vectors]
        
        # Query is the 5th vector (original float32)
        query_vector = db_vectors[5]
        results = dna_semantic_compression.dna_similarity_search_adc(query_vector, db_dna)
        
        # Top result should be index 5
        self.assertEqual(results[0][0], 5)
        # Distance should be low (not necessarily 0 because of quantization, but minimal)
        self.assertLess(results[0][1], 2.0)

if __name__ == '__main__':
    unittest.main()
