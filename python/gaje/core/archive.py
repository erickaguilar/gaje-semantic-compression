import json
import struct
import os


class GAJEArchive:
    """
    Unified File Format (.gaje)
    Structure:
    [MAGIC:4b][VER:2b][CODEBOOK_LEN:4b][CODEBOOK_JSON][DATA_COUNT:4b][DATA_ENTRIES...]
    """

    MAGIC = b"GAJE"
    VERSION = 2

    def __init__(self, codebook=None):
        self.codebook = codebook
        self.entries = []  # List of (label, dna_strand)

    def add(self, label, dna_strand):
        self.entries.append((label, dna_strand))

    def save(self, file_path):
        with open(file_path, "wb") as f:
            # Header
            f.write(self.MAGIC)
            f.write(struct.pack("H", self.VERSION))

            # Codebook
            cb_json = json.dumps(self.codebook).encode("utf-8")
            f.write(struct.pack("I", len(cb_json)))
            f.write(cb_json)

            # Data
            f.write(struct.pack("I", len(self.entries)))
            for label, dna in self.entries:
                label_b = label.encode("utf-8")
                f.write(struct.pack("I", len(label_b)))
                f.write(label_b)
                f.write(struct.pack("I", len(dna)))
                f.write(dna)
        print(f"📦 Archive saved: {file_path} ({os.path.getsize(file_path)} bytes)")

    @classmethod
    def load(cls, file_path):
        with open(file_path, "rb") as f:
            magic = f.read(4)
            if magic != cls.MAGIC:
                raise ValueError("Not a valid GAJE file")

            struct.unpack("H", f.read(2))[0]
            cb_len = struct.unpack("I", f.read(4))[0]
            codebook = json.loads(f.read(cb_len).decode("utf-8"))

            archive = cls(codebook)
            count = struct.unpack("I", f.read(4))[0]
            for _ in range(count):
                l_len = struct.unpack("I", f.read(4))[0]
                label = f.read(l_len).decode("utf-8")
                d_len = struct.unpack("I", f.read(4))[0]
                dna = f.read(d_len)
                archive.add(label, dna)
            return archive
