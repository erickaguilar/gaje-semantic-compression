import http.server
import socketserver
import json
import os
import sys
import platform

SERVER_DIR = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
PROJECT_ROOT = os.path.abspath(os.path.join(SERVER_DIR, "..", "..", ".."))
DIRECTORY = SERVER_DIR
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.core import _impl as dna_semantic_compression  # noqa: E402
from gaje.nn.stabilized import GenomicLLM  # noqa: E402

PORT = 8080
import threading  # noqa: E402

# Cache de modelos y lock para evitar cargas duplicadas o fallos por concurrencia
loaded_models = {}
model_lock = threading.Lock()


def get_model(model_name):
    with model_lock:
        if model_name in loaded_models:
            return loaded_models[model_name]

        model_dir = os.path.join(PROJECT_ROOT, "models")
        model_path = None
        
        # Search recursively in PROJECT_ROOT/models
        for root, _, files in os.walk(model_dir):
            if model_name in files:
                model_path = os.path.join(root, model_name)
                break

        if not model_path:
            print(
                f"❌ No se encontró el archivo de modelo '{model_name}' en {model_dir}"
            )
            return None

        print(f"🧬 Cargando modelo real: {model_path}")
        try:
            llm = GenomicLLM.load_genomic(os.path.abspath(model_path))
            llm.rust_llm.set_k_wta_ratio(0.0)
            loaded_models[model_name] = llm
            return llm
        except Exception as e:
            import traceback

            print(f"❌ Error cargando modelo {model_name}: {e}")
            traceback.print_exc()
            return None


class GajeHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_GET(self):
        if self.path == "/api/models":
            models_root = os.path.join(PROJECT_ROOT, "models")
            models = []
            seen_models = set()

            if os.path.exists(models_root):
                for root, _, files in os.walk(models_root):
                    for f in files:
                        if (f.endswith(".gaje") or f.endswith(".flat")) and f not in seen_models:
                            fpath = os.path.join(root, f)
                            mtime = os.path.getmtime(fpath)
                            from datetime import datetime

                            date_str = datetime.fromtimestamp(mtime).strftime(
                                "%Y-%m-%d %H:%M"
                            )
                            models.append({"name": f, "date": date_str})
                            seen_models.add(f)

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"models": models}).encode())
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/load_model":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data)
                model_name = data.get("model", "")

                print(f"🧬 Pre-cargando modelo: {model_name}...")
                llm = get_model(model_name)
                if not llm:
                    self.send_response(500)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps({"error": f"No se pudo cargar {model_name}"}).encode()
                    )
                    return

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps({"status": "ok", "model": model_name}).encode()
                )
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())

        elif self.path == "/api/chat":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data)

                message = data.get("message", "")
                model_name = data.get("model", "")

                print(f"[*] Procesando mensaje con modelo: {model_name}")

                llm = get_model(model_name)
                if not llm:
                    response_data = {
                        "error": f"Modelo {model_name} no disponible o error al cargar."
                    }
                    self.send_response(500)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(response_data).encode())
                    return

                # 1. Obtener Embedding (simulado del input para visualización)
                try:
                    tokens = llm.tokenizer.encode(message, add_special_tokens=False)
                    if hasattr(tokens, "ids"):
                        tokens = tokens.ids
                except Exception:
                    tokens = [0]

                # 2. Generación Genómica (con plantilla ChatML para modelos Instruct)
                import time

                formatted_message = message
                if not message.startswith("<|im_start|>"):
                    formatted_message = f"<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"

                start_time = time.time()
                response_text = ""
                tokens_count = 0
                print(f"[*] Generando respuesta para: {repr(formatted_message[:40])}...")
                try:
                    for token_text in llm.generate(
                        formatted_message,
                        max_new_tokens=60,
                        temperature=0.3,
                        repetition_penalty=1.1,
                    ):
                        if "<|im_end|>" in token_text:
                            token_text = token_text.replace("<|im_end|>", "").strip()
                            response_text += token_text
                            break
                        response_text += token_text
                        tokens_count += 1
                        if len(response_text) > 400:
                            break
                except Exception as e:
                    response_text = f"Error en generación: {e}"

                gen_time_ms = round((time.time() - start_time) * 1000, 2)
                tok_per_sec = (
                    round(tokens_count / (gen_time_ms / 1000), 2)
                    if gen_time_ms > 0
                    else 0
                )

                # 3. Visualización del primer token (para la UI de ADN)
                try:
                    first_token_id = tokens[0] if tokens else 0
                    emb_obj = getattr(llm, "embeddings", getattr(llm.rust_llm, "embeddings", None))
                    n_embd_val = getattr(llm, "n_embd", getattr(llm.rust_llm, "n_embd", 896))
                    if emb_obj and hasattr(emb_obj, "get_row"):
                        emb_row = emb_obj.get_row(first_token_id)
                        if hasattr(emb_row, "tolist"):
                            emb_row = emb_row.tolist()
                    else:
                        import numpy as np

                        emb_row = np.random.randn(n_embd_val).tolist()

                    thresholds = [-0.34, 0.0, 0.34]
                    dna_strand_bytes = dna_semantic_compression.quantize_embedding(
                        emb_row, thresholds
                    )

                    mapping = {0b00: "A", 0b01: "C", 0b11: "G", 0b10: "T"}
                    bases = []
                    for byte in dna_strand_bytes[:32]:
                        for shift in [6, 4, 2, 0]:
                            val = (byte >> shift) & 0b11
                            bases.append(mapping[val])

                    dna_visual = "".join(bases)
                except Exception as ex_dna:
                    print(f"⚠️ Warning visualizando ADN: {ex_dna}")
                    dna_visual = "ACGT" * 8

                # 5. Métricas e Info del Sistema (SF & HD)
                dims = getattr(llm, "n_embd", getattr(llm.rust_llm, "n_embd", 896))
                orig_size = dims * 4
                dna_size = (dims + 3) // 4
                ratio = orig_size / dna_size
                saved = (1 - (dna_size / orig_size)) * 100

                # Detallar Software (SF) y Hardware (HD)
                sf_info = (
                    f"Rust 2021 (NEON/SIMD) + PyO3 / Python {platform.python_version()}"
                )
                cpu_name = platform.processor() or platform.machine()
                try:
                    if os.path.exists("/proc/cpuinfo"):
                        with open("/proc/cpuinfo", "r") as f:
                            for line in f:
                                if "model name" in line:
                                    cpu_name = line.split(":")[1].strip()
                                    break
                except Exception:
                    pass

                hd_info = f"{cpu_name} ({platform.machine()}) - Native CPU"

                response_data = {
                    "response": response_text or "Procesamiento completado.",
                    "dna": dna_visual,
                    "metrics": {
                        "dims": dims,
                        "original_size": orig_size,
                        "dna_size": dna_size,
                        "ratio": ratio,
                        "saved": saved,
                        "latency_ms": gen_time_ms,
                        "tokens_sec": tok_per_sec,
                        "tokens_count": tokens_count,
                        "sf_info": sf_info,
                        "hd_info": hd_info,
                    },
                }

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode())
            except Exception as exc:
                import traceback

                print(f"❌ Error fatal en do_POST: {exc}")
                traceback.print_exc()
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(exc)}).encode())
        else:
            self.send_error(404)


class ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    os.chdir(DIRECTORY)

    with ThreadingTCPServer(("", PORT), GajeHandler) as httpd:
        print(f"🚀 Servidor GAJE Visual Real activo en http://localhost:{PORT}")
        print("Presiona Ctrl+C para detener.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServidor detenido.")
            httpd.server_close()
