import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PYTHON_DIR = os.path.join(PROJECT_ROOT, "python")
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

from gaje.download import (
    resolve_model_key,
    format_bytes,
    get_system_specs,
    MODEL_REGISTRY,
)


class TestModelDownloader(unittest.TestCase):
    def test_format_bytes(self):
        self.assertEqual(format_bytes(500), "500 B")
        self.assertEqual(format_bytes(1024), "1.0 KB")
        self.assertEqual(format_bytes(1024 * 1024 * 500), "500.0 MB")
        self.assertEqual(format_bytes(1024 * 1024 * 1024 * 2), "2.00 GB")

    def test_resolve_model_key(self):
        self.assertEqual(resolve_model_key("nano"), "gaje_nano_1.5b")
        self.assertEqual(resolve_model_key("1.5b"), "gaje_nano_1.5b")
        self.assertEqual(resolve_model_key("gaje_nano_1.5b"), "gaje_nano_1.5b")
        self.assertEqual(resolve_model_key("prime"), "gaje_prime_3b")
        self.assertEqual(resolve_model_key("ultra"), "gaje_ultra_7b")
        self.assertEqual(resolve_model_key("deepseek"), "deepseek_r1_1.5b")
        self.assertEqual(resolve_model_key("smol"), "smollm2_135m_gguf")
        self.assertIsNone(resolve_model_key("nonexistent_model_xyz"))

    def test_get_system_specs(self):
        specs = get_system_specs()
        self.assertIn("is_android", specs)
        self.assertIn("machine", specs)
        self.assertIn("total_ram_gb", specs)
        self.assertIn("avail_ram_gb", specs)
        self.assertIn("free_storage_gb", specs)
        self.assertGreater(specs["total_ram_gb"], 0.0)

    def test_model_registry_validity(self):
        for key, info in MODEL_REGISTRY.items():
            self.assertIn("filename", info)
            self.assertIn("description", info)
            self.assertIn("size_mb", info)
            self.assertIn("min_ram_gb", info)
            self.assertIn("url", info)
            self.assertGreater(info["size_mb"], 0)
            self.assertGreater(info["min_ram_gb"], 0)


if __name__ == "__main__":
    unittest.main()
