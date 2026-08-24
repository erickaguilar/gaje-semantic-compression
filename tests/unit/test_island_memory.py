"""Unit tests for GAJE Island Model (.gmem) Memory Manager."""

import os
import sys
import tempfile
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "examples", "ui", "web_ui"))

from gaje.processing.island_memory import IslandMemoryManager
from prompt_templates import format_prompt


class TestIslandMemoryManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.gmem_path = os.path.join(self.temp_dir.name, "test_island.gmem")
        self.mgr = IslandMemoryManager(self.gmem_path, dim=128)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_add_and_retrieve_memory(self):
        self.mgr.add_memory("episodic", "El usuario prefiere respuestas en español técnico.")
        self.mgr.add_memory("documental", "El motor GAJE utiliza compresión de 4 bits Q4_0.")
        self.mgr.add_memory("conversational", "Usuario: Hola | Asistente: ¡Hola!")

        # Buscar por similitud
        results = self.mgr.retrieve_context("idioma de preferencia y español", top_k=1)
        self.assertGreater(len(results), 0)
        entry, sim = results[0]
        self.assertEqual(entry.niche, "episodic")
        self.assertIn("español", entry.text)

    def test_gmem_binary_roundtrip(self):
        self.mgr.add_memory("documental", "Test memory item 12345")
        self.mgr.save()

        # Recargar desde archivo .gmem
        reloaded = IslandMemoryManager(self.gmem_path, dim=128)
        self.assertEqual(len(reloaded.entries), len(self.mgr.entries))
        self.assertTrue(any("12345" in e.text for e in reloaded.entries))

    def test_prompt_injection_within_budget(self):
        injection = self.mgr.format_memory_injection("cuéntame sobre la compresión Q4_0 y GAJE")
        self.assertIsNotNone(injection)
        self.assertIn("Memoria de Largo Plazo", injection)

        formatted_prompt = format_prompt(
            model_name="qwen2_5_3b.flat",
            message="¿Qué es GAJE?",
            island_context=injection,
        )
        self.assertIn("Memoria de Largo Plazo", formatted_prompt)
        self.assertIn("GAJE", formatted_prompt)


if __name__ == "__main__":
    unittest.main()
