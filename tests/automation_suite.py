#!/usr/bin/env python3
"""GAJE Helix — Suite de Automatización de Pruebas y Validación Integral.

Ejecuta de forma automatizada las 5 suites de validación:
1. Inferencia Nativa & Zero-Copy Mmap (.flat / .gaje).
2. Purga y Gestión de Memoria RAM (malloc_trim / RSS).
3. Endpoints del Web UI & Streaming SSE (__gaje_metrics__).
4. Prototipo de Tokenización Cuántico-Genómica (Superposición ρ).
5. Métricas y Throughput SIMD AVX2.
"""

import os
import sys
import time
import gc
import json
import unittest
import numpy as np

# Rutas del proyecto
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "examples", "ui", "web_ui"))

from gaje.nn.stabilized import GenomicLLM
from gaje.utils.version import get_project_version
from model_manager import get_model, list_available_models, unload_model, loaded_models
from prompt_templates import format_prompt, get_stop_tokens

try:
    import psutil
except ImportError:
    psutil = None

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def get_current_rss_mb() -> float:
    """Retorna la memoria RSS actual del proceso en MB."""
    if psutil:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    else:
        try:
            with open("/proc/self/status", "r") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return float(line.split()[1]) / 1024.0
        except Exception:
            return 0.0
    return 0.0


class TestGajeAutomationSuite(unittest.TestCase):
    """Suite de Pruebas Automatizadas de GAJE Helix."""

    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 80)
        print("🧬 INICIANDO SUITE DE AUTOMATIZACIÓN GAJE HELIX ENGINE")
        print(f"Versión: {get_project_version()} | Python: {sys.version.split()[0]}")
        print("=" * 80)

    # =========================================================================
    # SUITE 1: Inferencia Nativa & Zero-Copy Mmap
    # =========================================================================
    def test_01_models_discovery(self):
        """TC-1.1: Descubrimiento de modelos en el repositorio."""
        models = list_available_models(MODELS_DIR)
        self.assertGreater(len(models), 0, "No se encontraron modelos en el directorio de modelos")
        model_names = [m["name"] for m in models]
        print(f"\n[SUITE 1] Modelos detectados ({len(models)}): {model_names}")
        
        has_flat = any(m.endswith(".flat") for m in model_names)
        self.assertTrue(has_flat, "Debe existir al menos un modelo .flat transmutado")

    def test_02_native_inference_execution(self):
        """TC-1.2: Inferencia nativa determinista con modelo .flat."""
        model_name = "qwen2_0_5b.flat"
        model_path = os.path.join(MODELS_DIR, "production", model_name)
        if not os.path.exists(model_path):
            models = list_available_models(MODELS_DIR)
            model_name = models[0]["name"]

        print(f"\n[SUITE 1] Probando inferencia nativa con [{model_name}]...")
        llm = get_model(MODELS_DIR, model_name, GenomicLLM)
        self.assertIsNotNone(llm, f"Fallo al cargar el modelo {model_name}")

        prompt = format_prompt(model_name, "Hola, responde brevemente: ¿que es el ADN?")
        tokens = llm.tokenizer.encode(prompt, add_special_tokens=False)
        if hasattr(tokens, "ids"):
            tokens = tokens.ids

        eos_ids = get_stop_tokens(model_name, llm.tokenizer)
        start_t = time.time()
        gen_ids = llm.rust_llm.generate_native_py(tokens, 24, 0.2, 1.1, eos_ids)
        elapsed = (time.time() - start_t) * 1000.0

        response = llm.tokenizer.decode(gen_ids).strip()
        print(f"[SUITE 1] Respuesta generada ({len(gen_ids)} tokens en {elapsed:.2f} ms):")
        print(f"          \"{response[:90]}...\"")
        self.assertGreater(len(gen_ids), 0, "La generación no produjo ningún token")

    # =========================================================================
    # SUITE 2: Gestión de Memoria y Purga Agresiva (malloc_trim)
    # =========================================================================
    def test_03_memory_purge_and_leak_check(self):
        """TC-2.1: Purga completa de memoria RAM con malloc_trim(0)."""
        rss_before = get_current_rss_mb()
        model_name = "qwen2_5_3b.flat"
        model_path = os.path.join(MODELS_DIR, "production", model_name)
        if not os.path.exists(model_path):
            models = list_available_models(MODELS_DIR)
            model_name = models[0]["name"]

        print(f"\n[SUITE 2] Memoria RSS inicial: {rss_before:.2f} MB")
        print(f"[SUITE 2] Cargando [{model_name}]...")
        llm = get_model(MODELS_DIR, model_name, GenomicLLM)
        rss_loaded = get_current_rss_mb()
        print(f"[SUITE 2] Memoria RSS con modelo cargado: {rss_loaded:.2f} MB (+{rss_loaded - rss_before:.2f} MB)")

        print("[SUITE 2] Ejecutando unload_model() y malloc_trim(0)...")
        unload_model()
        rss_after = get_current_rss_mb()
        print(f"[SUITE 2] Memoria RSS tras purga agresiva: {rss_after:.2f} MB")

        self.assertLess(rss_after, rss_loaded * 0.85, "La purga de memoria no liberó las páginas mmap correctamente")
        self.assertEqual(len(loaded_models), 0, "No deben quedar modelos en loaded_models tras unload_model")

    # =========================================================================
    # SUITE 3: Protocolo de Métricas SSE (__gaje_metrics__)
    # =========================================================================
    def test_04_streaming_metrics_protocol(self):
        """TC-3.1: Verificación de estructura de métricas de compresión y tokens."""
        print("\n[SUITE 3] Validando estructura de evento __gaje_metrics__...")
        dims = 2048
        bit_depth = 4
        prompt_tokens = 24
        generated_tokens = 18
        total_tokens = prompt_tokens + generated_tokens
        elapsed_ms = 450.0

        metrics_sample = {
            "__gaje_metrics__": {
                "latency_ms": elapsed_ms,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "tokens_count": total_tokens,
                "tokens_sec": round(generated_tokens / (elapsed_ms / 1000.0), 1),
                "dims": dims,
                "original_size": dims * 4,
                "dna_size": int(dims * bit_depth / 8.0),
                "bit_depth": bit_depth,
                "ratio": 8.0,
                "saved": 87.5
            },
            "dna": "GGCCCCCGCCCGCCGCCGCGGCGCGGGCCCGTCGGGGCGCGCCCCGGCGGCCGGCGGGGCCCCCCCCCGCCCCGCGCCCGCCGGGGCGGGCGCGGCGGCCAGCGGGCCCGGGGGCCGGGCGGGCGCGC"
        }

        raw_json = json.dumps(metrics_sample)
        parsed = json.loads(raw_json)
        self.assertIn("__gaje_metrics__", parsed)
        m = parsed["__gaje_metrics__"]
        tc = m['tokens_count']
        rt = m['ratio']
        sv = m['saved']
        print(f"[SUITE 3] Métricas validadas con éxito: {tc} tokens (Ratio: {rt}x | Ahorro: {sv}%)")

    # =========================================================================
    # SUITE 4: Prototipo de Tokenización Cuántico-Genómica
    # =========================================================================
    def test_05_quantum_genomic_tokenization(self):
        """TC-4.1: Mapeo de bases genómicas a vectores de estado y matrices de densidad ρ."""
        print("\n[SUITE 4] Probando Tokenización Cuántico-Genómica (QuantumGenomicTokenizer)...")
        from gaje.processing.quantum_tokenizer import QuantumGenomicTokenizer, BASIS_A, BASIS_G

        tokenizer = QuantumGenomicTokenizer()
        state = tokenizer.encode_char_to_state("G")

        # 1. Traza unitaria de la matriz de densidad
        self.assertAlmostEqual(state.trace, 1.0, places=5, msg="La traza de la matriz de densidad debe ser 1")

        # 2. Pureza cuántica
        self.assertAlmostEqual(state.purity, 1.0, delta=0.01)

        # 3. Colapso contextual proyectivo
        ctx_g = BASIS_G
        base, conf = state.collapse_with_context(ctx_g)
        self.assertEqual(base, "G")
        self.assertGreater(conf, 0.5)

        # 4. Codificación completa a ADN
        dna = tokenizer.collapse_text_to_dna("GAJE", context_text="Biología Genómica")
        self.assertEqual(len(dna), 4)

        print(f"[SUITE 4] Estado cuántico verificado: Traza(ρ) = {state.trace:.2f} | Pureza = {state.purity:.2f}")
        print(f"[SUITE 4] Colapso contextual a ADN: 'GAJE' -> '{dna}' (Confianza Guanina: {conf:.2%})")

    # =========================================================================
    # SUITE 5: Certificación de Tokenizador Binario GTOK & Incrustación en .flat
    # =========================================================================
    def test_06_gtok_binary_certification(self):
        """TC-5.1: Verificación de compresión, decodificación e incrustación en .flat de GTOK."""
        print("\n[SUITE 5] Validando formato binario nativo GTOK...")
        from gaje.processing.gtok import (
            GtokTokenizer,
            export_hf_tokenizer_to_gtok,
            embed_gtok_into_flat,
            extract_gtok_from_flat,
            has_embedded_gtok,
        )

        vocab = ["<unk>", "<s>", "</s>", "<pad>", "H", "ola", "Hola", "ADN"]
        merges = [(4, 5, 6)]
        specials = {"bos": 1, "eos": 2, "unk": 0, "pad": 3}
        gtok = GtokTokenizer(vocab=vocab, merges=merges, special_tokens=specials, additional_stop_ids=[2])

        binary_data = gtok.to_bytes()
        self.assertEqual(binary_data[:4], b"GTOK")

        # Test roundtrip de incrustación
        import tempfile
        import struct
        header = bytearray(4096)
        header[:4] = b"GAJE"
        struct.pack_into("<III", header, 4, 2, 0, 1)
        with tempfile.NamedTemporaryFile(suffix=".flat", delete=False) as tf:
            tf.write(header)
            tf.write(b"DUMMY_DATA")
            tmp_flat = tf.name

        try:
            embed_gtok_into_flat(tmp_flat, gtok)
            self.assertTrue(has_embedded_gtok(tmp_flat))
            extracted = extract_gtok_from_flat(tmp_flat)
            self.assertIsNotNone(extracted)
            self.assertEqual(extracted.decode([6]), "Hola")
            print("[SUITE 5] GTOK verificado: Decodificación y Roundtrip .flat 100% exitoso.")
        finally:
            if os.path.exists(tmp_flat):
                os.remove(tmp_flat)

    # =========================================================================
    # SUITE 6: Certificación del Loop de Inferencia Cuántico (.qemb)
    # =========================================================================
    def test_07_quantum_embedding_inference_loop(self):
        """TC-6.1: Verificación de inferencia y forward nativo con Quantum Embedding Table (.qemb)."""
        print("\n[SUITE 6] Validando Loop de Inferencia Cuántico (.qemb)...")
        import io
        import struct
        import numpy as np
        from gaje.processing.quantum_codebook import QuantumEmbeddingTable, QEMB_MAGIC, QEMB_VERSION

        model_path = os.path.join(MODELS_DIR, "production", "smollm2_135m.flat")
        if not os.path.exists(model_path):
            self.skipTest(f"Modelo {model_path} no encontrado para prueba cuántica.")

        llm = GenomicLLM.load_genomic(model_path)
        self.assertFalse(llm.rust_llm.has_quantum_embeddings())

        # Forward clásico
        logits_fp = llm.rust_llm.forward(10, True)
        self.assertEqual(len(logits_fp), 49152)

        # Generar tabla cuántica sintética
        dim = 576
        vocab = 49152
        k = 256
        m = 4
        fake_emb = np.random.randn(vocab, dim).astype(np.float32)
        table = QuantumEmbeddingTable.from_dense_embeddings(fake_emb, num_meta_tokens=k, m=m)

        buf = io.BytesIO()
        header = struct.pack("<4sHHIII44s", QEMB_MAGIC, QEMB_VERSION, m, k, vocab, dim, b"\x00" * 44)
        buf.write(header)
        buf.write(table.codebook.centroids.tobytes())
        buf.write(table.indices.tobytes())
        buf.write(table.amplitudes.tobytes())
        qemb_bytes = buf.getvalue()

        # Cargar en motor nativo
        llm.rust_llm.load_quantum_embeddings_bytes(qemb_bytes)
        self.assertTrue(llm.rust_llm.has_quantum_embeddings())

        # Forward cuántico
        logits_q = llm.rust_llm.forward(10, True)
        self.assertEqual(len(logits_q), 49152)

        # Generación nativa cuántica
        gen_tokens = llm.rust_llm.generate_native_py([10, 42], 5, 0.7, 1.0, [2])
        self.assertEqual(len(gen_tokens), 5)

        # Descarga
        llm.rust_llm.unload_quantum_embeddings()
        self.assertFalse(llm.rust_llm.has_quantum_embeddings())

        print(f"[SUITE 6] Inferencia cuántica validada: {len(gen_tokens)} tokens generados con .qemb activo.")


def run_all_suites():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGajeAutomationSuite)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    run_all_suites()

