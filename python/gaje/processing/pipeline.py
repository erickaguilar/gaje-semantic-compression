import os
import json
from gaje.nn.stabilized import GenomicLayer, GenomicTransformerBlock, GenomicLLM


class DatasetProcessor:
    """
    Procesador de datasets para entrenamiento continuo y nacimiento genómico.
    Soporta formatos planos (.txt) y estructurados (.jsonl).
    """

    @staticmethod
    def load_dataset(file_path: str) -> list[str]:
        """
        Carga un dataset dependiendo de su extensión.
        Retorna una lista de strings listos para ser procesados por el tokenizer.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset no encontrado: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".jsonl":
            return DatasetProcessor._load_jsonl(file_path)
        elif ext == ".txt":
            return DatasetProcessor._load_txt(file_path)
        else:
            raise ValueError(f"Formato no soportado: {ext}. Use .txt o .jsonl")

    @staticmethod
    def _load_txt(file_path: str) -> list[str]:
        """Carga un archivo de texto plano línea por línea."""
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Filtramos líneas muy cortas o vacías
        dataset = [line.strip() for line in lines if len(line.strip()) > 5]
        print(f"[*] Procesados {len(dataset)} fragmentos de texto plano.")
        return dataset

    @staticmethod
    def _load_jsonl(file_path: str) -> list[str]:
        """
        Carga un archivo JSONL asumiendo formato de instrucción:
        {"instruction": "...", "response": "..."}
        Opcionalmente soporta "system" o "context".
        """
        dataset = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)

                    # Extraer campos (soporta varios formatos comunes)
                    instruction = data.get(
                        "instruction", data.get("prompt", data.get("user", ""))
                    )
                    response = data.get(
                        "response", data.get("completion", data.get("assistant", ""))
                    )
                    system = data.get("system", data.get("context", ""))

                    if not instruction or not response:
                        print(
                            f"[!] Advertencia: Línea {line_idx+1} en JSONL ignorada (faltan campos 'instruction' o 'response')."
                        )
                        continue

                    # Formatear como conversación
                    formatted_text = ""
                    if system:
                        formatted_text += f"Sistema: {system}\n"
                    formatted_text += f"Usuario: {instruction}\nAsistente: {response}"

                    dataset.append(formatted_text)

                except json.JSONDecodeError:
                    print(
                        f"[!] Advertencia: Línea {line_idx+1} en JSONL no es un JSON válido."
                    )

        print(f"[*] Procesadas {len(dataset)} interacciones estructuradas (JSONL).")
        return dataset


# Este archivo se mantiene por compatibilidad con benchmarks previos
# pero ahora redirige al motor estabilizado v0.6.0 para evitar errores de Q8

__all__ = ["GenomicLayer", "GenomicTransformerBlock", "GenomicLLM", "DatasetProcessor"]
