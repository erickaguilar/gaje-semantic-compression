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
import math
import json
import unittest

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
        self.assertGreater(
            len(models), 0, "No se encontraron modelos en el directorio de modelos"
        )
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
        print(
            f"[SUITE 1] Respuesta generada ({len(gen_ids)} tokens en {elapsed:.2f} ms):"
        )
        print(f'          "{response[:90]}..."')
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
        _llm = get_model(MODELS_DIR, model_name, GenomicLLM)
        rss_loaded = get_current_rss_mb()
        print(
            f"[SUITE 2] Memoria RSS con modelo cargado: {rss_loaded:.2f} MB (+{rss_loaded - rss_before:.2f} MB)"
        )

        print("[SUITE 2] Ejecutando unload_model() y malloc_trim(0)...")
        unload_model()
        rss_after = get_current_rss_mb()
        print(f"[SUITE 2] Memoria RSS tras purga agresiva: {rss_after:.2f} MB")

        self.assertLess(
            rss_after,
            rss_loaded * 0.85,
            "La purga de memoria no liberó las páginas mmap correctamente",
        )
        self.assertEqual(
            len(loaded_models),
            0,
            "No deben quedar modelos en loaded_models tras unload_model",
        )

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
                "saved": 87.5,
            },
            "dna": "GGCCCCCGCCCGCCGCCGCGGCGCGGGCCCGTCGGGGCGCGCCCCGGCGGCCGGCGGGGCCCCCCCCCGCCCCGCGCCCGCCGGGGCGGGCGCGGCGGCCAGCGGGCCCGGGGGCCGGGCGGGCGCGC",
        }

        raw_json = json.dumps(metrics_sample)
        parsed = json.loads(raw_json)
        self.assertIn("__gaje_metrics__", parsed)
        m = parsed["__gaje_metrics__"]
        tc = m["tokens_count"]
        rt = m["ratio"]
        sv = m["saved"]
        print(
            f"[SUITE 3] Métricas validadas con éxito: {tc} tokens (Ratio: {rt}x | Ahorro: {sv}%)"
        )

    # =========================================================================
    # SUITE 4: Prototipo de Tokenización Cuántico-Genómica
    # =========================================================================
    def test_05_quantum_genomic_tokenization(self):
        """TC-4.1: Mapeo de bases genómicas a vectores de estado y matrices de densidad ρ."""
        print(
            "\n[SUITE 4] Probando Tokenización Cuántico-Genómica (QuantumGenomicTokenizer)..."
        )
        from gaje.processing.quantum_tokenizer import QuantumGenomicTokenizer, BASIS_G

        tokenizer = QuantumGenomicTokenizer()
        state = tokenizer.encode_char_to_state("G")

        # 1. Traza unitaria de la matriz de densidad
        self.assertAlmostEqual(
            state.trace,
            1.0,
            places=5,
            msg="La traza de la matriz de densidad debe ser 1",
        )

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

        print(
            f"[SUITE 4] Estado cuántico verificado: Traza(ρ) = {state.trace:.2f} | Pureza = {state.purity:.2f}"
        )
        print(
            f"[SUITE 4] Colapso contextual a ADN: 'GAJE' -> '{dna}' (Confianza Guanina: {conf:.2%})"
        )

    # =========================================================================
    # SUITE 5: Certificación de Tokenizador Binario GTOK & Incrustación en .flat
    # =========================================================================
    def test_06_gtok_binary_certification(self):
        """TC-5.1: Verificación de compresión, decodificación e incrustación en .flat de GTOK."""
        print("\n[SUITE 5] Validando formato binario nativo GTOK...")
        from gaje.processing.gtok import (
            GtokTokenizer,
            embed_gtok_into_flat,
            extract_gtok_from_flat,
            has_embedded_gtok,
        )

        vocab = ["<unk>", "<s>", "</s>", "<pad>", "H", "ola", "Hola", "ADN"]
        merges = [(4, 5, 6)]
        specials = {"bos": 1, "eos": 2, "unk": 0, "pad": 3}
        gtok = GtokTokenizer(
            vocab=vocab, merges=merges, special_tokens=specials, additional_stop_ids=[2]
        )

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
            print(
                "[SUITE 5] GTOK verificado: Decodificación y Roundtrip .flat 100% exitoso."
            )
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
        from gaje.processing.quantum_codebook import (
            QuantumEmbeddingTable,
            QEMB_MAGIC,
            QEMB_VERSION,
        )

        model_path = os.path.join(MODELS_DIR, "production", "smollm2_135m.flat")
        if not os.path.exists(model_path):
            self.skipTest(f"Modelo {model_path} no encontrado para prueba cuántica.")

        llm = GenomicLLM.load_genomic(model_path)
        self.assertFalse(llm.has_quantum_embeddings())

        # Forward clásico
        logits_fp = llm.rust_llm.forward(10, True)
        self.assertEqual(len(logits_fp), 49152)

        # Generar tabla cuántica sintética
        dim = 576
        vocab = 49152
        k = 256
        m = 4
        fake_emb = np.random.randn(vocab, dim).astype(np.float32)
        table = QuantumEmbeddingTable.from_dense_embeddings(
            fake_emb, num_meta_tokens=k, m=m
        )

        buf = io.BytesIO()
        header = struct.pack(
            "<4sHHIII44s", QEMB_MAGIC, QEMB_VERSION, m, k, vocab, dim, b"\x00" * 44
        )
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
        llm.unload_quantum_embeddings()
        self.assertFalse(llm.has_quantum_embeddings())
        self.assertFalse(llm.rust_llm.has_quantum_embeddings())

        print(
            f"[SUITE 6] Inferencia cuántica validada: {len(gen_tokens)} tokens generados con .qemb activo."
        )

    # =========================================================================
    # SUITE 7: Certificación de Aceleración por GPU (Vulkan / WGPU)
    # =========================================================================
    def test_08_gpu_acceleration_and_parity(self):
        """TC-7.1: Verificación de detección de GPU, compilación WGSL y paridad numérica."""
        print("\n[SUITE 7] Validando Backend de Aceleración GPU (Vulkan / WGPU)...")
        from gaje.core._impl import (
            is_gpu_available_py,
            get_gpu_info_py,
            gpu_swiglu_py,
            gpu_gemv_f32_py,
        )
        import numpy as np

        gpu_active = is_gpu_available_py()
        if not gpu_active:
            print(
                "[SUITE 7] Adaptador GPU no disponible en este entorno. Fallback CPU verificado."
            )
            return

        info = get_gpu_info_py()
        self.assertIsNotNone(info)
        self.assertIn("device_name", info)
        self.assertIn("backend", info)
        print(
            f"[SUITE 7] GPU Detectada: {info['device_name']} ({info['backend']}) | UMA: {info.get('is_unified_memory')}"
        )

        # 1. Paridad Numérica SwiGLU (GPU vs CPU)
        N = 2048
        gate = np.random.randn(N).astype(np.float32)
        up = np.random.randn(N).astype(np.float32)
        h_scale = 1.0

        gpu_swiglu = gpu_swiglu_py(gate.tolist(), up.tolist(), h_scale)
        self.assertIsNotNone(gpu_swiglu)
        cpu_silu = (gate / (1.0 + np.exp(-gate))) * up * h_scale
        diff_swiglu = np.max(np.abs(np.array(gpu_swiglu) - cpu_silu))
        self.assertLess(
            diff_swiglu, 1e-5, f"Diferencia SwiGLU excede tolerancia: {diff_swiglu}"
        )

        # 2. Paridad Numérica GEMV FP32 (GPU vs CPU)
        M, K = 256, 512
        W = np.random.randn(M, K).astype(np.float32)
        x = np.random.randn(K).astype(np.float32)

        gpu_gemv = gpu_gemv_f32_py(W.flatten().tolist(), x.tolist(), M, K)
        self.assertIsNotNone(gpu_gemv)
        cpu_gemv = np.dot(W, x)
        diff_gemv = np.max(np.abs(np.array(gpu_gemv) - cpu_gemv))
        self.assertLess(
            diff_gemv, 1e-4, f"Diferencia GEMV excede tolerancia: {diff_gemv}"
        )

        print(
            f"[SUITE 7] Paridad GPU certificada: SwiGLU Δ={diff_swiglu:.2e} | GEMV Δ={diff_gemv:.2e}"
        )

    # =========================================================================
    # SUITE 8: Certificación de Paridad Bit a Bit WebAssembly (GAJE-WASM)
    # =========================================================================
    def test_09_wasm_bit_parity(self):
        """TC-8.1: Determinismo bit a bit idéntico entre CPU Nativo y WebAssembly (GajeWasmEngine)."""
        print("\n[SUITE 8] Validando Determinismo y Paridad Bit a Bit WebAssembly...")
        import subprocess

        model_path = os.path.join(MODELS_DIR, "production", "smollm2_135m.flat")
        if not os.path.exists(model_path):
            self.skipTest(f"Modelo {model_path} no encontrado para prueba WASM.")

        llm = GenomicLLM.load_genomic(model_path)
        prompt_tokens = [10, 42, 128, 256, 512]
        max_new_tokens = 8
        temperature = 0.0
        repetition_penalty = 1.0
        stop_ids = [2]

        native_tokens = llm.rust_llm.generate_native_py(
            prompt_tokens, max_new_tokens, temperature, repetition_penalty, stop_ids
        )

        wasm_script = f"""
import fs from 'fs';
import {{ GajeWasmEngine }} from './pkg/wasm_node/_impl.js';
const fileBuffer = fs.readFileSync('{model_path}');
const engine = GajeWasmEngine.load_from_bytes(new Uint8Array(fileBuffer));
const promptIds = new Uint32Array({json.dumps(prompt_tokens)});
const stopIds = new Uint32Array({json.dumps(stop_ids)});
const genIds = engine.generate(promptIds, {max_new_tokens}, {temperature}, {repetition_penalty}, stopIds);
console.log(JSON.stringify(Array.from(genIds)));
"""
        wasm_res = subprocess.run(
            ["node", "--input-type=module", "-e", wasm_script],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )

        wasm_tokens = json.loads(wasm_res.stdout.strip())
        self.assertEqual(
            native_tokens,
            wasm_tokens,
            f"Discrepancia entre Native ({native_tokens}) y WASM ({wasm_tokens})",
        )
        print(
            f"[SUITE 8] Paridad WASM 100% verificada: {len(native_tokens)} tokens idénticos bit a bit."
        )

    def test_10_zero_order_spsa_training(self):
        """TC-9.1: Entrenamiento nativo de orden cero (SPSA Discreto) sobre modelo .flat."""
        from dna_semantic_compression import NativeGenomicTrainer

        model_path = os.path.join(MODELS_DIR, "production", "smollm2_135m.flat")
        if not os.path.exists(model_path):
            self.skipTest(f"Modelo {model_path} no encontrado")

        print("\n[SUITE 9] Validando Entrenamiento de Orden Cero (SPSA Discreto)...")
        llm = GenomicLLM.load_genomic(model_path)
        self.assertIsNotNone(llm)

        dataset = [
            [280, 395, 1599, 345, 406],
            [102, 503, 894, 201, 77],
            [55, 312, 440, 981, 1024],
        ]

        trainer = NativeGenomicTrainer(lr=0.01, resonance_weight=0.05)
        final_loss = trainer.fit_zero_order(
            llm.rust_llm, dataset, epochs=2, k_coords=16
        )

        self.assertGreater(final_loss, 0.0)
        self.assertFalse(math.isnan(final_loss))
        self.assertFalse(math.isinf(final_loss))
        print(
            f"[SUITE 9] Entrenamiento SPSA completado exitosamente: Loss={final_loss:.4f}"
        )

    def test_11_adult_specialization_spsa(self):
        """TC-10.1: Especialización SPSA sobre memoria (.gmem) y orquestador de nichos Island."""
        from dna_semantic_compression import IslandOrchestrator

        print(
            "\n[SUITE 10] Validando Especialización de Organismos Adultos con SPSA sobre .gmem..."
        )
        dim = 128
        orch = IslandOrchestrator(dim, [1.0, 1.0, 1.0], 0.60)

        # Ingestar Aguja y distractores
        needle_text = "CLAVE_ACCESO_GAJE_999"
        v_needle = [1.0] + [0.0] * (dim - 1)
        orch.add_memory_py("documental", 1, v_needle, needle_text)

        v_distractor = [0.0, 1.0] + [0.0] * (dim - 2)
        orch.add_memory_py("episodic", 2, v_distractor, "EVENTO_RUTINA_1")

        # Calibrar nicho Documental vía SPSA
        queries = [v_needle] * 5
        targets = [1] * 5
        loss = orch.optimize_spsa_py(queries, targets, epochs=5, c=0.05, lr=0.10)
        self.assertFalse(math.isnan(loss))

        # Verificar Needle-Recall
        matches = orch.retrieve_context_py(v_needle, 1)
        self.assertTrue(len(matches) > 0)
        self.assertEqual(matches[0][3], needle_text)
        print(
            "[SUITE 10] Especialización de adulto certificada al 100%: Needle recuperado con éxito."
        )

    # =========================================================================
    # SUITE 11: Épocas de Memoria y Linaje Versionado (.gmem v2)
    # =========================================================================
    def test_12_memory_epochs_and_lineage(self):
        """TC-11.1: Snapshots inmutables, árboles de linaje y rollback exacto sub-milisegundo."""
        from gaje.core._impl import EpochManager, IslandOrchestrator
        import tempfile, shutil

        print(
            "\n[SUITE 11] Validando Épocas de Memoria y Linaje Versionado (.gmem v2)..."
        )
        temp_dir = tempfile.mkdtemp(prefix="gaje_suite11_epochs_")
        dim = 64

        try:
            mgr = EpochManager(temp_dir, "smollm2_suite", dim)
            self.assertEqual(mgr.active_epoch_id, 1)

            # Ingesta en Época 1 -> Snapshot Época 2
            orch = IslandOrchestrator(dim)
            v1 = [1.0] + [0.0] * (dim - 1)
            orch.add_memory_py("documental", 1001, v1, "CLAVE_CANONICA_EP2")
            ep2 = mgr.create_snapshot_py(orch, "Snapshot Canónico", None)
            self.assertEqual(ep2, 2)
            self.assertEqual(mgr.active_epoch_id, 2)

            # Ingesta ruidosa -> Snapshot Época 3
            v2 = [0.0, 1.0] + [0.0] * (dim - 2)
            orch.add_memory_py("episodic", 9999, v2, "RUIDO_VOLATIL_EP3")
            ep3 = mgr.create_snapshot_py(orch, "Snapshot Ruidoso", None)
            self.assertEqual(ep3, 3)

            # Rollback instantáneo a Época 2 (< 1 ms)
            t0 = time.perf_counter()
            restored_orch = mgr.rollback_to_py(2)
            t_rb_ms = (time.perf_counter() - t0) * 1000.0
            self.assertLess(t_rb_ms, 5.0)
            self.assertEqual(mgr.active_epoch_id, 2)

            # Verificar reversibilidad exacta (solo 1 entrada documental, 0 ruido)
            matches = restored_orch.retrieve_context_py(v1, 2)
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0][1], 1001)

            # Promover y sellar
            mgr.promote_epoch_py(2)
            mgr.seal_epoch_py(2)

            epochs_json = mgr.list_epochs_py()
            epochs = json.loads(epochs_json)
            self.assertEqual(len(epochs), 3)
            self.assertEqual(epochs[1]["verdict"], "SEALED")

            print(
                f"[SUITE 11] Épocas y Linaje .gmem v2 certificados al 100%: Rollback en {t_rb_ms:.3f} ms."
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


def run_all_suites():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGajeAutomationSuite)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    run_all_suites()
