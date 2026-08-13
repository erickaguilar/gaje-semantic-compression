"""GAJE-Flow Visual Web UI Server.

Modular HTTP server for local LLM inference, embedding visualization,
and Island Model context orchestration.
"""

import http.server
import json
import os
import platform
import socketserver
import sys
import time

SERVER_DIR = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
PROJECT_ROOT = os.path.abspath(os.path.join(SERVER_DIR, "..", "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))
sys.path.insert(0, SERVER_DIR)

from gaje.nn.stabilized import GenomicLLM  # noqa: E402
from gaje.utils.version import get_project_version  # noqa: E402
from model_manager import get_model, list_available_models  # noqa: E402
from prompt_templates import format_prompt, get_stop_tokens  # noqa: E402

PORT = 8080
MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")

# Configuración central del Island Model (.gmem). Fuente única de verdad
# para la UI; no se duplica en el HTML.
ISLAND_CONFIG = {
    "memory_type": ".gmem (Zero-Copy)",
    "retrieval_latency_ms": 0.75,
    "context_budget": 512,
    "pills": ["⚡ Episódica", "📚 Documental", "💬 Conversación"],
}


def _model_quality(name: str) -> float:
    """Estima los parámetros del modelo (en miles de millones) para ordenar por calidad."""
    n = name.lower()
    if "3b" in n:
        return 3.0
    if "1_5b" in n:
        return 1.5
    if "0_5b" in n:
        return 0.5
    if "smollm" in n or "135" in n:
        return 0.135
    return 0.0


def _detect_simd() -> str:
    """Detecta los flags SIMD reales de la CPU desde /proc/cpuinfo (Linux)."""
    flags = []
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("flags"):
                    flags = line.split(":", 1)[1].split()
                    break
    except OSError:
        return platform.machine().lower() in ("aarch64", "arm64") and "NEON" or "SIMD"
    mapping = [
        ("avx512f", "AVX-512"),
        ("avx2", "AVX2"),
        ("fma", "FMA"),
        ("avx", "AVX"),
        ("sse4_2", "SSE4.2"),
        ("asimd", "NEON"),
        ("sve", "SVE"),
    ]
    present = [label for flag, label in mapping if flag in flags]
    return "/".join(present) if present else "SIMD genérico"


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def get_runtime_info() -> dict:
    """Información real del entorno de ejecución (arquitectura, CPU, SIMD)."""
    arch = platform.machine()
    cpu = _cpu_model()
    simd = _detect_simd()
    cores = os.cpu_count() or 1
    py = sys.version.split()[0]
    return {
        "engine_version": get_project_version(),
        "python_version": py,
        "architecture": arch,
        "cpu": cpu,
        "cores": cores,
        "simd": simd,
        "os": f"{platform.system()} {platform.release()}",
        "software": f"Rust 2021 ({simd}) + PyO3 / Python {py}",
        "hardware": f"{cpu} - {arch} ({cores} cores)",
        "island": ISLAND_CONFIG,
    }


class GajeHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=SERVER_DIR, **kwargs)

    def do_GET(self):
        if self.path == "/api/models":
            models = list_available_models(MODELS_ROOT)
            models.sort(key=lambda m: _model_quality(m.get("name", "")), reverse=True)
            self._send_json({"models": models})
        elif self.path == "/api/info":
            self._send_json(get_runtime_info())
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/load_model":
            self._handle_load_model()
        elif self.path == "/api/chat":
            self._handle_chat()
        else:
            self.send_error(404, "Endpoint not found")

    def _handle_load_model(self):
        try:
            data = self._read_json_body()
            model_name = data.get("model", "")
            print(f"🧬 Pre-cargando modelo: {model_name}...")

            llm = get_model(MODELS_ROOT, model_name, GenomicLLM)
            if not llm:
                self._send_json(
                    {"error": f"No se pudo cargar {model_name}"}, status=500
                )
                return

            self._send_json({"status": "ok", "model": model_name})
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)

    def _handle_chat(self):
        try:
            data = self._read_json_body()
            message = data.get("message", "")
            model_name = data.get("model", "")
            _runtime = get_runtime_info()

            print(f"[*] Procesando mensaje con modelo: {model_name}")
            llm = get_model(MODELS_ROOT, model_name, GenomicLLM)
            if not llm:
                self._send_json(
                    {"error": f"Modelo {model_name} no disponible."}, status=500
                )
                return

            # 1. Formatear Prompt según Arquitectura
            formatted_message = format_prompt(model_name, message)
            tokens = llm.tokenizer.encode(formatted_message, add_special_tokens=False)
            if hasattr(tokens, "ids"):
                tokens = tokens.ids

            # 2. Inferencia Nativa
            start_time = time.time()
            eos_ids = get_stop_tokens(model_name, llm.tokenizer)

            try:
                # Use stable, low-entropy sampling (temperature=0.2, rep_penalty=1.1) to avoid loops
                # and factual hallucinations in highly compressed models.
                gen_ids = llm.rust_llm.generate_native_py(
                    tokens, 512, 0.2, 1.1, eos_ids
                )
            except Exception as e:
                print(f"⚠️ Warning en generate_native_py: {e}")
                gen_ids = [2]

            elapsed_ms = (time.time() - start_time) * 1000.0

            # 3. Decodificar Respuesta
            full_response = llm.tokenizer.decode(gen_ids)
            cleaned_response = (
                full_response.split("<|im_end|>")[0]
                .split("<|endoftext|>")[0]
                .split("<end_of_turn>")[0]
                .strip()
            )

            num_tokens = len(gen_ids)
            tok_per_sec = (
                (num_tokens / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0.0
            )

            # 4. Simulación de DNA / Metadatos para Visualización Web UI
            dna_sample = "GGCCCCCGCCCGCCGCCGCGGCGCGGGCCCGTCGGGGCGCGCCCCGGCGGCCGGCGGGGCCCCCCCCCGCCCCGCGCCCGCCGGGGCGGGCGCGGCGGCCAGCGGGCCCGGGGGCCGGGCGGGCGCGC"

            dims = getattr(llm, "n_embd", 576)
            if callable(dims):
                dims = dims()

            bit_depth = getattr(llm, "bit_depth", 4)
            if bit_depth == 32:
                ratio = 1.0
                saved = 0.0
                compressed_size = dims * 4
            else:
                ratio = 32.0 / bit_depth
                saved = 100.0 * (1.0 - (bit_depth / 32.0))
                compressed_size = int(dims * bit_depth / 8.0)

            response_data = {
                "response": cleaned_response,
                "metrics": {
                    "latency_ms": round(elapsed_ms, 2),
                    "tokens_count": num_tokens,
                    "tokens_sec": round(tok_per_sec, 2),
                    "dims": dims,
                    "original_size": dims * 4,
                    "dna_size": compressed_size,
                    "bit_depth": bit_depth,
                    "ratio": round(ratio, 1),
                    "saved": round(saved, 2),
                    "sf_info": _runtime["software"],
                    "hd_info": _runtime["hardware"],
                },
                "dna": dna_sample,
            }
            self._send_json(response_data)

        except Exception as e:
            import traceback

            traceback.print_exc()
            self._send_json({"error": str(e)}, status=500)

    def _read_json_body(self) -> dict:
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)
        return json.loads(post_data)

    def _send_json(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))


if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("", PORT), GajeHandler) as httpd:
        httpd.daemon_threads = True
        print(f"🚀 Servidor GAJE Visual Real activo en http://localhost:{PORT}")
        print("Presiona Ctrl+C para detener.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Servidor detenido.")
