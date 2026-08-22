"""Unit tests for GAJE GTOK zero-dependency binary tokenizer."""

import os
import struct
import tempfile
import unittest
from python.gaje.processing.gtok import (
    GtokTokenizer,
    export_hf_tokenizer_to_gtok,
    embed_gtok_into_flat,
    extract_gtok_from_flat,
    has_embedded_gtok,
)


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

    def test_embed_and_extract_from_synthetic_flat(self):
        # 1. Crear un modelo .flat sintético válido (cabecera GAJE de 4096 bytes + weights dummy)
        header = bytearray(4096)
        header[:4] = b"GAJE"
        struct.pack_into("<III", header, 4, 2, 0, 1)  # version=2, flags=0, num_tensors=1
        dummy_weights = b"WEIGHTS_DUMMY_DATA_12345678"

        with tempfile.NamedTemporaryFile(suffix=".flat", delete=False) as tf:
            tf.write(header)
            tf.write(dummy_weights)
            synthetic_flat_path = tf.name

        try:
            self.assertFalse(has_embedded_gtok(synthetic_flat_path))

            # 2. Tokenizador a incrustar
            vocab = ["<unk>", "<s>", "</s>", "H", "ola", "Hola"]
            merges = [(3, 4, 5)]
            specials = {"bos": 1, "eos": 2, "unk": 0, "pad": 0}
            gtok = GtokTokenizer(
                vocab=vocab,
                merges=merges,
                special_tokens=specials,
                additional_stop_ids=[2],
            )

            # 3. Incrustar
            embed_gtok_into_flat(synthetic_flat_path, gtok)
            self.assertTrue(has_embedded_gtok(synthetic_flat_path))

            # 4. Extraer y verificar
            extracted = extract_gtok_from_flat(synthetic_flat_path)
            self.assertIsNotNone(extracted)
            self.assertEqual(len(extracted.vocab), len(vocab))
            self.assertEqual(extracted.decode([5]), "Hola")
        finally:
            if os.path.exists(synthetic_flat_path):
                os.remove(synthetic_flat_path)


if __name__ == "__main__":
    unittest.main()
