import json
import struct
import os


class GAJEArchive:
    """
    Unified File Format (.gaje) v3
    Structure:
    [MAGIC:4b][VER:2b][CB_LEN:4b][CB_JSON][EPI_CB_LEN:4b][EPI_CB_JSON][DATA_COUNT:4b][ENTRIES...]
    Each Entry: [L_LEN:4b][LABEL][DNA_LEN:4b][DNA][EPI_DNA_LEN:4b][EPI_DNA]
    """

    MAGIC = b"GAJE"
    VERSION = 3

    def __init__(self, codebook=None, epigenetic_codebook=None):
        self.codebook = codebook
        self.epigenetic_codebook = epigenetic_codebook
        self.entries = []  # List of (label, dna, epi_dna)

    def add(self, label, dna_strand, epigenetic_strand=None):
        self.entries.append((label, dna_strand, epigenetic_strand))

    def save(self, file_path):
        with open(file_path, "wb") as f:
            # Header
            f.write(self.MAGIC)
            f.write(struct.pack("H", self.VERSION))

            # Base Codebook
            cb_json = json.dumps(self.codebook).encode("utf-8")
            f.write(struct.pack("I", len(cb_json)))
            f.write(cb_json)
            
            # Epigenetic Codebook
            epi_cb_json = json.dumps(self.epigenetic_codebook).encode("utf-8") if self.epigenetic_codebook else b"{}"
            f.write(struct.pack("I", len(epi_cb_json)))
            f.write(epi_cb_json)

            # Data
            f.write(struct.pack("I", len(self.entries)))
            for label, dna, epi_dna in self.entries:
                label_b = label.encode("utf-8")
                f.write(struct.pack("I", len(label_b)))
                f.write(label_b)
                
                f.write(struct.pack("I", len(dna)))
                f.write(dna)
                
                epi_dna_b = epi_dna if epi_dna else b""
                f.write(struct.pack("I", len(epi_dna_b)))
                f.write(epi_dna_b)
        print(f"📦 Archive saved: {file_path} ({os.path.getsize(file_path)} bytes)")

    @classmethod
    def load(cls, file_path):
        with open(file_path, "rb") as f:
            magic = f.read(4)
            if magic != cls.MAGIC:
                raise ValueError("Not a valid GAJE file")

            ver = struct.unpack("H", f.read(2))[0]
            
            # Base Codebook
            cb_len = struct.unpack("I", f.read(4))[0]
            codebook = json.loads(f.read(cb_len).decode("utf-8"))
            
            epi_codebook = None
            if ver >= 3:
                epi_cb_len = struct.unpack("I", f.read(4))[0]
                epi_codebook = json.loads(f.read(epi_cb_len).decode("utf-8"))

            archive = cls(codebook, epi_codebook)
            count = struct.unpack("I", f.read(4))[0]
            for _ in range(count):
                l_len = struct.unpack("I", f.read(4))[0]
                label = f.read(l_len).decode("utf-8")
                
                d_len = struct.unpack("I", f.read(4))[0]
                dna = f.read(d_len)
                
                epi_dna = None
                if ver >= 3:
                    e_len = struct.unpack("I", f.read(4))[0]
                    epi_dna = f.read(e_len)
                    if not epi_dna: epi_dna = None
                
                archive.add(label, dna, epi_dna)
            return archive
