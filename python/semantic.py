import struct
import numpy as np

class GenomicCodonMapper:
    """
    Biological-inspired text compression.
    Mapps high-frequency semantic markers to short binary codons.
    """
    def __init__(self):
        self.codon_map = {
            "ai": b"\x00",
            "memory": b"\x01",
            "system": b"\x02",
            "error": b"\x03",
        }

    def encode(self, text):
        # Simplified codon mapping logic
        tokens = text.lower().split()
        encoded = bytearray()
        for token in tokens:
            if token in self.codon_map:
                encoded.extend(self.codon_map[token])
            else:
                encoded.extend(token.encode('utf-8') + b" ")
        return bytes(encoded)

class DNASemanticRecord:
    def __init__(self, engine):
        self.engine = engine
        self.mapper = GenomicCodonMapper()

    def pack(self, text, embedding):
        """
        Creates a 'Semantic Chromosome' (Packed BLOB).
        Structure: [HEADER:4b][TEXT_LEN:4b][TEXT_DATA][EMBEDDING_DATA]
        """
        dna_text = self.mapper.encode(text)
        dna_vector = self.engine.quantize_embedding(embedding)
        
        header = b"DNA\x02" # Magic number + version
        text_len = struct.pack("I", len(dna_text))
        
        return header + text_len + dna_text + bytes(dna_vector)
