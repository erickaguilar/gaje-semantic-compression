import os
import unittest
import subprocess
import sys

# Asegurar que el código local esté en el path
sys.path.insert(0, os.path.abspath("python"))

from gaje.nn.stabilized import GenomicLLM


class TestGoldEmbryo(unittest.TestCase):
    MODEL_PATH = "models/checkpoints/gold_embryo.gaje"

    def setUp(self):
        # Limpiar si existe
        if os.path.exists(self.MODEL_PATH):
            os.remove(self.MODEL_PATH)

    def test_hatch_and_load(self):
        """Valida que el Embrión de Oro nazca con las dimensiones y peso correctos."""

        # 1. Ejecutar el script de incubación (Paso 4 del plan)
        # Esto fallará si el script no existe aún (Red Phase)
        script_path = "scripts/hatch_gold_embryo.py"
        result = subprocess.run(
            [sys.executable, script_path], capture_output=True, text=True
        )

        self.assertEqual(
            result.returncode, 0, f"El script de incubación falló: {result.stderr}"
        )
        self.assertTrue(
            os.path.exists(self.MODEL_PATH), "El archivo del modelo no fue creado."
        )

        # 2. Validar Peso (Meta: < 10 MB)
        file_size = os.path.getsize(self.MODEL_PATH)
        max_size = 10 * 1024 * 1024  # 10 MB
        self.assertLess(
            file_size,
            max_size,
            f"El modelo excede los 10MB: {file_size / 1024 / 1024:.2f} MB",
        )
        print(f"✅ Peso del Embrión: {file_size / 1024 / 1024:.2f} MB")

        # 3. Validar Carga e Integridad
        model = GenomicLLM.load_genomic(self.MODEL_PATH)

        self.assertEqual(
            model.n_blocks, 8, "El modelo debe tener exactamente 8 bloques."
        )
        self.assertEqual(model.n_embd, 384, "La dimensión oculta debe ser 384.")

        # 4. Validar Vocabulario
        # El vocabulario real depende del tokenizador, pero el SDD dice ~16k
        self.assertLessEqual(model.rust_llm.embeddings.out_features, 17000)

        print("✅ Integridad estructural validada.")


if __name__ == "__main__":
    unittest.main()
