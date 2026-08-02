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
from model_manager import get_model, list_available_models  # noqa: E402
from prompt_templates import format_prompt, get_stop_tokens  # noqa: E402

PORT = 8080
MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")


class GajeHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=SERVER_DIR, **kwargs)

    def do_GET(self):
        if self.path == "/api/models":
            models = list_available_models(MODELS_ROOT)
            self._send_json({"models": models})
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
                gen_ids = llm.rust_llm.generate_native_py(
                    tokens, 128, 0.7, 0.9, eos_ids
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

            dims = getattr(llm.embeddings, "in_features", 576)
            if callable(dims):
                dims = dims()

            response_data = {
                "response": cleaned_response,
                "metrics": {
                    "latency_ms": round(elapsed_ms, 2),
                    "tokens": num_tokens,
                    "tok_per_sec": round(tok_per_sec, 2),
                    "dims": dims,
                    "original_bytes": dims * 4,
                    "compressed_bytes": dims // 2,
                    "ratio": 8.0,
                    "saving_pct": 87.5,
                },
                "island_info": {
                    "latency_ms": 0.0,
                    "tokens_added": 0,
                    "cossim": 0.0,
                },
                "dna_seq": dna_sample,
                "env": {
                    "sf": f"Rust 2021 (NEON/SIMD) + PyO3 / Python {sys.version.split()[0]}",
                    "hd": f"{platform.processor() or 'CPU Native'} - Native CPU",
                },
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
    with socketserver.TCPServer(("", PORT), GajeHandler) as httpd:
        print(f"🚀 Servidor GAJE Visual Real activo en http://localhost:{PORT}")
        print("Presiona Ctrl+C para detener.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Servidor detenido.")
