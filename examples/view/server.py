import http.server
import socketserver
import json
import os
import numpy as np
from gaje.core import _impl as dna_semantic_compression

PORT = 8080
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# Intentar cargar SentenceTransformer para embeddings reales
try:
    from sentence_transformers import SentenceTransformer
    print("[*] Cargando modelo de lenguaje (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    HAS_MODEL = True
except ImportError:
    print("[!] SentenceTransformer no encontrado. Usando vectores sintéticos.")
    HAS_MODEL = False

class GajeHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_GET(self):
        if self.path == '/api/models':
            checkpoints_dir = os.path.join("models", "checkpoints")
            models = []
            if os.path.exists(checkpoints_dir):
                for f in os.listdir(checkpoints_dir):
                    if f.endswith('.gaje'):
                        fpath = os.path.join(checkpoints_dir, f)
                        mtime = os.path.getmtime(fpath)
                        from datetime import datetime
                        date_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
                        models.append({"name": f, "date": date_str})
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"models": models}).encode())
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == '/api/chat':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)
            
            message = data.get('message', '')
            model_name = data.get('model', 'auto')
            
            print(f"[*] Procesando mensaje con modelo: {model_name}")

            # 1. Obtener Embedding
            if HAS_MODEL:
                embedding = model.encode(message).tolist()
            else:
                np.random.seed(sum(ord(c) for c in message) % 1234)
                embedding = np.random.normal(0, 1, 384).tolist()

            dims = len(embedding)
            
            # 2. Simular carga de pesos .gaje si no es 'auto'
            if model_name != "auto":
                model_path = os.path.join("models/checkpoints", model_name)
                if os.path.exists(model_path):
                    print(f"[+] Pesos cargados desde: {model_path}")
                else:
                    print(f"[!] Archivo no encontrado: {model_path}")

            # 3. Cuantizar con el Motor Rust
            thresholds = [-0.34, 0.0, 0.34]
            dna_strand_bytes = dna_semantic_compression.quantize_embedding(embedding, thresholds)
            
            # 4. Convertir a Bases (A, C, G, T)
            mapping = {0b00: "A", 0b01: "C", 0b11: "G", 0b10: "T"}
            bases = []
            for byte in dna_strand_bytes[:32]:
                for shift in [6, 4, 2, 0]:
                    val = (byte >> shift) & 0b11
                    bases.append(mapping[val])
            
            dna_visual = "".join(bases)
            
            # 5. Calcular Métricas
            orig_size = dims * 4
            dna_size = len(dna_strand_bytes)
            ratio = orig_size / dna_size
            saved = (1 - (dna_size / orig_size)) * 100
            
            # 6. Respuesta Dinámica según el modelo
            if "smollm2" in model_name:
                bot_msg = f"[SmolLM2]: Procesando token genómico. Densidad optimizada para {model_name}."
            elif "polyglot" in model_name:
                bot_msg = f"[Polyglot]: Análisis multilingüe completado. Resonancia detectada."
            elif "english" in model_name:
                bot_msg = f"[English-Org]: Semantic refinement applied using {model_name}."
            else:
                bot_msg = "Entrada semántica procesada con éxito por el núcleo GAJE."
            
            response_data = {
                "response": bot_msg,
                "dna": dna_visual,
                "metrics": {
                    "dims": dims,
                    "original_size": orig_size,
                    "dna_size": dna_size,
                    "ratio": ratio,
                    "saved": saved
                }
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode())
        else:
            self.send_error(404)

if __name__ == "__main__":
    os.chdir(DIRECTORY)
    with socketserver.TCPServer(("", PORT), GajeHandler) as httpd:
        print(f"🚀 Servidor GAJE Visual activo en http://localhost:{PORT}")
        print("Presiona Ctrl+C para detener.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServidor detenido.")
            httpd.server_close()
