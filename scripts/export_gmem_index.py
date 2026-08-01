#!/usr/bin/env python3
"""
🧬 Exportador de Índices de Memoria Persistente GAJE (.gmem)
-------------------------------------------------------------
Genera un archivo binario .gmem a partir de documentos de texto plano
o texto ingresado para su uso en el Island Model.
"""

import sys
import os
import struct

# Script independiente de generación binaria .gmem


def create_gmem_file(output_path, text_entries, dim=128):
    """
    Crea un archivo binario .gmem alineado a 64 bytes.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # 1. Armar Header de 64 bytes
    # magic: 4B, version: 4B, dim: 4B, index_type: 1B, _pad: 3B, num_entries: 8B, reserved: 40B
    magic = b"GMEM"
    version = 1
    index_type = 0
    pad = b"\x00" * 3
    num_entries = len(text_entries)
    reserved = b"\x00" * 40

    header = struct.pack(
        "<4sIIB3sQ40s",
        magic,
        version,
        dim,
        index_type,
        pad,
        num_entries,
        reserved,
    )
    assert len(header) == 64, f"Header size is {len(header)}, expected 64"

    with open(output_path, "wb") as f:
        f.write(header)

        for idx, text in enumerate(text_entries):
            # Generar vector proyectado DNI determinista (dim * float32)
            vector = [0.0] * dim
            vector[idx % dim] = 1.0

            # Escribir ID (u64)
            f.write(struct.pack("<Q", idx + 1000))

            # Escribir Vector (dim * f32)
            for val in vector:
                f.write(struct.pack("<f", val))

            # Escribir Text (u32 length + utf8 bytes)
            text_bytes = text.encode("utf-8")
            f.write(struct.pack("<I", len(text_bytes)))
            f.write(text_bytes)

    print(
        f"✅ Archivo de memoria persistente generado exitosamente: {output_path} ({os.path.getsize(output_path)} bytes)"
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Uso: python scripts/export_gmem_index.py <archivo_salida.gmem> <texto_o_archivo>"
        )
        print(
            "Ejemplo: python scripts/export_gmem_index.py data/knowledge.gmem 'GAJE es un motor semántico nativo'"
        )
        sys.exit(1)

    out_path = sys.argv[1]
    input_src = sys.argv[2]

    entries = []
    if os.path.exists(input_src):
        with open(input_src, "r", encoding="utf-8") as f:
            entries = [line.strip() for line in f if line.strip()]
    else:
        entries = [input_src]

    create_gmem_file(out_path, entries)
