"""Unit tests for GAJE GTOK zero-dependency binary tokenizer."""

import os
import unittest
from python.gaje.processing.gtok import GtokTokenizer, export_hf_tokenizer_to_gtok

class TestGtokTokenizer(unittest.TestCase):
    def test_synthetic_gtok_roundtrip(self):
        vocab = ["<unk>", "<s>", "</s>", "A", "B", "C", "AB", "BC", "ABC"]
        merges = [(3, 4, 6), (4, 5, 7), (6, 5, 8)]
        specials = {"bos": 1, "eos": 2, "unk": 0, "pad": 0}
        extra_stops = [2]

        gtok = GtokTokenizer(
            vocab=vocab,
            merges=merges,
            special_tokens=specials,
            additional_stop_ids=extra_stops,
        )

        binary_data = gtok.to_bytes()
        self.assertGreater(len(binary_data), 36)
        self.assertEqual(binary_data[:4], b"GTOK")

        # Reload from bytes
        reloaded = GtokTokenizer.from_bytes(binary_data)
        self.assertEqual(len(reloaded.vocab), len(vocab))
        self.assertEqual(len(reloaded.merges), len(merges))
        self.assertEqual(reloaded.special_tokens["eos"], 2)

        # Test decode
        decoded = reloaded.decode([3, 4, 5])
        self.assertEqual(decoded, "ABC")

    def test_hf_conversion_if_exists(self):
        hf_json = "models/core/tokenizer.json"
        if os.path.exists(hf_json):
            out_gtok = "models/core/tokenizer.gtok"
            gtok = export_hf_tokenizer_to_gtok(hf_json, out_gtok)
            self.assertTrue(os.path.exists(out_gtok))
            self.assertGreater(len(gtok.vocab), 1000)

if __name__ == "__main__":
    unittest.main()
