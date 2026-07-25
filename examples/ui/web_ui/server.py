import http.server
import socketserver
import json
import os
import sys

# Asegurar que usamos el código local de 'python/'
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.core import _impl as dna_semantic_compression
from gaje.nn.stabilized import GenomicLLM

PORT = 8080
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# Cache de modelos para no recargar en cada mensaje
loaded_models = {}


def get_model(model_name):
    if model_name in loaded_models:
        return loaded_models[model_name]

    # Buscar en múltiples ubicaciones
    possible_paths = [
        os.path.join(PROJECT_ROOT, "models", model_name),
        os.path.join(PROJECT_ROOT, "models", "checkpoints", model_name),
        os.path.join(PROJECT_ROOT, "models", "archive", model_name),
    ]

    model_path = None
    for p in possible_paths:
        if os.path.exists(p):
            model_path = p
            break

    if not model_path:
        return None

    print(f"🧬 Cargando modelo real: {model_path}")
    try:
        llm = GenomicLLM.load_genomic(model_path)
        loaded_models[model_name] = llm
        return llm
    except Exception as e:
        print(f"❌ Error cargando modelo {model_name}: {e}")
        return None


class GajeHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_GET(self):
        if self.path == "/api/models":
            search_dirs = [
                os.path.join(PROJECT_ROOT, "models"),
                os.path.join(PROJECT_ROOT, "models", "checkpoints"),
                os.path.join(PROJECT_ROOT, "models", "archive"),
            ]
            models = []
            seen_models = set()

            for d in search_dirs:
                if os.path.exists(d):
                    for f in os.listdir(d):
                        if f.endswith(".gaje") and f not in seen_models:
                            fpath = os.path.join(d, f)
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
        if self.path == "/api/chat":
            content_length = int(self.headers["Content-Length"])
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
            except:
                tokens = [0]

            # 2. Generación Genómica
            response_text = ""
            print("[*] Generando respuesta...")
            try:
                # Usamos un número pequeño de tokens para la web por latencia
                # Aumentamos el repetition_penalty para evitar bucles
                for token_text in llm.generate(
                    message, max_new_tokens=50, temperature=0.6, repetition_penalty=1.2
                ):
                    response_text += token_text
                    if len(response_text) > 400:
                        break  # Límite de seguridad
            except Exception as e:
                response_text = f"Error en generación: {e}"

            # 3. Visualización del primer token (para la UI de ADN)
            first_token_id = tokens[0] if tokens else 0
            # Recuperar embedding del primer token
            emb_row = llm.embeddings.get_row(first_token_id)

            # 4. Cuantizar para visualización (Motor Rust)
            thresholds = [-0.34, 0.0, 0.34]
            dna_strand_bytes = dna_semantic_compression.quantize_embedding(
                emb_row.tolist(), thresholds
            )

            # Convertir a Bases (A, C, G, T)
            mapping = {0b00: "A", 0b01: "C", 0b11: "G", 0b10: "T"}
            bases = []
            for byte in dna_strand_bytes[:32]:
                for shift in [6, 4, 2, 0]:
                    val = (byte >> shift) & 0b11
                    bases.append(mapping[val])

            dna_visual = "".join(bases)

            # 5. Métricas
            dims = llm.n_embd
            orig_size = dims * 4
            dna_size = (dims + 3) // 4
            ratio = orig_size / dna_size
            saved = (1 - (dna_size / orig_size)) * 100

            response_data = {
                "response": response_text or "Procesamiento completado.",
                "dna": dna_visual,
                "metrics": {
                    "dims": dims,
                    "original_size": orig_size,
                    "dna_size": dna_size,
                    "ratio": ratio,
                    "saved": saved,
                },
            }

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode())
        else:
            self.send_error(404)


if __name__ == "__main__":
    os.chdir(DIRECTORY)
    with socketserver.TCPServer(("", PORT), GajeHandler) as httpd:
        print(f"🚀 Servidor GAJE Visual Real activo en http://localhost:{PORT}")
        print("Presiona Ctrl+C para detener.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServidor detenido.")
            httpd.server_close()
