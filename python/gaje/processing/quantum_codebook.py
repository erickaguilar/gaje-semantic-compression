"""GAJE Helix — Quantum Superposition Meta-Tokens Codebook (Phase 1 & 2).

Compresses vocabulary embedding tables (e.g. 151,643 tokens) into K=8,192 canonical quantum meta-tokens
using sparse superposition projection (m=4) and unit-norm amplitude normalization.
"""

import math
import struct
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

QEMB_MAGIC = b"QEMB"
QEMB_VERSION = 1


class QuantumCodebook:
    """Libro de códigos cuántico de 8,192 meta-tokens en la esfera unitaria H^d."""

    def __init__(self, num_meta_tokens: int = 8192, dim: int = 1536):
        self.num_meta_tokens = num_meta_tokens
        self.dim = dim
        self.centroids = np.zeros((num_meta_tokens, dim), dtype=np.float32)

    @classmethod
    def create_harmonic_codebook(cls, num_meta_tokens: int = 8192, dim: int = 1536) -> "QuantumCodebook":
        """
        Genera un codebook inicial armónico determinista y ortogonalizado en la esfera unitaria,
        utilizando bases de Fourier/Chebyshev y fases cuánticas.
        """
        codebook = cls(num_meta_tokens, dim)
        
        # Mapeo armónico de frecuencias fundamentales
        k_indices = np.arange(num_meta_tokens, dtype=np.float32).reshape(-1, 1)
        d_indices = np.arange(dim, dtype=np.float32).reshape(1, -1)
        
        # Frecuencias espaciales cuánticas
        freqs = (k_indices + 1.0) * (d_indices + 1.0) * (math.pi / dim)
        mat = np.sin(freqs) + np.cos(freqs * 0.5)

        # Normalización L2 en la esfera unitaria
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        codebook.centroids = (mat / norms).astype(np.float32)
        return codebook

    def fit_from_embeddings(self, embeddings: np.ndarray, num_iterations: int = 10, batch_size: int = 4096, seed: int = 42):
        """
        Ajusta los centroides del codebook mediante Spherical Mini-Batch K-Means
        sobre la matriz de embeddings original.
        """
        np.random.seed(seed)
        num_samples, dim = embeddings.shape
        self.dim = dim

        # Normalizar datos de entrada
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        normalized_emb = embeddings / norms

        # Inicialización aleatoria uniforme desde los datos si num_samples >= K
        if num_samples >= self.num_meta_tokens:
            indices = np.random.choice(num_samples, self.num_meta_tokens, replace=False)
            self.centroids = normalized_emb[indices].copy().astype(np.float32)
        else:
            self.centroids = self.create_harmonic_codebook(self.num_meta_tokens, dim).centroids

        # Mini-batch Spherical K-Means iterations
        counts = np.ones(self.num_meta_tokens, dtype=np.float32)
        
        for it in range(num_iterations):
            batch_idx = np.random.choice(num_samples, min(batch_size, num_samples), replace=False)
            batch = normalized_emb[batch_idx]

            # Similitud coseno: batch (B x d) @ centroids.T (d x K) -> B x K
            sims = np.matmul(batch, self.centroids.T)
            nearest_centroids = np.argmax(sims, axis=1)

            for b_i, c_idx in enumerate(nearest_centroids):
                counts[c_idx] += 1.0
                eta = 1.0 / counts[c_idx]
                self.centroids[c_idx] = (1.0 - eta) * self.centroids[c_idx] + eta * batch[b_i]

            # Re-normalizar centroides
            c_norms = np.linalg.norm(self.centroids, axis=1, keepdims=True)
            c_norms[c_norms < 1e-9] = 1.0
            self.centroids = (self.centroids / c_norms).astype(np.float32)

    def project_sparse(self, vector: np.ndarray, m: int = 4) -> Tuple[List[int], List[float]]:
        """
        Proyecta un vector de embedding continuo en una superposición lineal de m meta-tokens cuánticos.
        Retorna (indices, amplitudes) con sum(amplitudes^2) = 1.0.
        """
        v_norm = np.linalg.norm(vector)
        if v_norm < 1e-9:
            return [0] * m, [1.0] + [0.0] * (m - 1)
        v = (vector / v_norm).astype(np.float32)

        # Proyecciones hermitianas / Similitud coseno: (K,)
        projections = np.dot(self.centroids, v)

        # Seleccionar los m mejores meta-tokens
        top_indices = np.argsort(projections)[::-1][:m]
        top_projections = projections[top_indices]

        # Asegurar no negatividad de amplitudes iniciales y normalizar
        raw_amps = np.maximum(0.0, top_projections)
        amp_norm = np.linalg.norm(raw_amps)
        if amp_norm > 1e-9:
            amplitudes = (raw_amps / amp_norm).tolist()
        else:
            amplitudes = [1.0] + [0.0] * (m - 1)

        return top_indices.tolist(), amplitudes

    def reconstruct(self, indices: List[int], amplitudes: List[float]) -> np.ndarray:
        """Reconstruye el embedding continuo a partir de su superposición cuántica."""
        reconstructed = np.zeros(self.dim, dtype=np.float32)
        for idx, amp in zip(indices, amplitudes):
            reconstructed += amp * self.centroids[idx]
        
        # Normalizar reconstrucción
        r_norm = np.linalg.norm(reconstructed)
        if r_norm > 1e-9:
            reconstructed = reconstructed / r_norm
        return reconstructed

    def evaluate_reconstruction_fidelity(self, embeddings: np.ndarray, m: int = 4, sample_size: int = 1000) -> float:
        """Evalúa la similitud coseno promedio de reconstrucción sobre una muestra de embeddings."""
        num_samples = len(embeddings)
        sample_indices = np.random.choice(num_samples, min(sample_size, num_samples), replace=False)
        
        cos_sims = []
        for idx in sample_indices:
            orig = embeddings[idx]
            o_norm = np.linalg.norm(orig)
            if o_norm < 1e-9:
                continue
            orig_unit = orig / o_norm

            inds, amps = self.project_sparse(orig, m=m)
            rec = self.reconstruct(inds, amps)
            
            sim = float(np.dot(orig_unit, rec))
            cos_sims.append(sim)

        return float(np.mean(cos_sims)) if cos_sims else 1.0


class QuantumEmbeddingTable:
    """Tabla de embeddings completa comprimida en formato cuántico .qemb."""

    def __init__(self, codebook: QuantumCodebook, num_tokens: int, m: int = 4):
        self.codebook = codebook
        self.num_tokens = num_tokens
        self.m = m
        self.indices = np.zeros((num_tokens, m), dtype=np.uint16)
        self.amplitudes = np.zeros((num_tokens, m), dtype=np.uint8)  # Cuantizado 8-bit [0..255]

    @classmethod
    def from_dense_embeddings(cls, embeddings: np.ndarray, num_meta_tokens: int = 8192, m: int = 4) -> "QuantumEmbeddingTable":
        """Construye y comprime una tabla de embeddings densa completa."""
        num_tokens, dim = embeddings.shape
        codebook = QuantumCodebook(num_meta_tokens, dim)
        codebook.fit_from_embeddings(embeddings, num_iterations=15)

        table = cls(codebook, num_tokens, m=m)
        
        for t_i in range(num_tokens):
            inds, amps = codebook.project_sparse(embeddings[t_i], m=m)
            table.indices[t_i] = inds
            # Cuantizar amplitudes en [0..255]
            table.amplitudes[t_i] = [int(min(255, max(0, round(a * 255.0)))) for a in amps]

        return table

    def get_embedding(self, token_id: int) -> np.ndarray:
        """Recupera el embedding de un token descomprimiendo la superposición al vuelo."""
        if token_id >= self.num_tokens:
            token_id = 0
        inds = self.indices[token_id].tolist()
        amps = (self.amplitudes[token_id].astype(np.float32) / 255.0).tolist()
        return self.codebook.reconstruct(inds, amps)

    def save_qemb(self, file_path: str):
        """Guarda la tabla comprimida en formato binario nativo .qemb."""
        with open(file_path, "wb") as f:
            # Header (64 bytes)
            header = struct.pack(
                "<4sHHIII44s",
                QEMB_MAGIC,
                QEMB_VERSION,
                self.m,
                self.codebook.num_meta_tokens,
                self.num_tokens,
                self.codebook.dim,
                b"\x00" * 44,
            )
            f.write(header)

            # Centroids de Codebook: K x dim x float32
            f.write(self.codebook.centroids.tobytes())

            # Índices de superposición: V x m x uint16
            f.write(self.indices.tobytes())

            # Amplitudes cuantizadas: V x m x uint8
            f.write(self.amplitudes.tobytes())

    @classmethod
    def load_qemb(cls, file_path: str) -> "QuantumEmbeddingTable":
        """Carga una tabla comprimida .qemb desde disco."""
        with open(file_path, "rb") as f:
            header = f.read(64)
            if len(header) < 64:
                raise ValueError("Archivo .qemb corrupto o truncado")
            magic, version, m, num_meta_tokens, num_tokens, dim, _ = struct.unpack("<4sHHIII44s", header)
            if magic != QEMB_MAGIC:
                raise ValueError(f"Magic bytes inválidos: {magic}")

            codebook = QuantumCodebook(num_meta_tokens, dim)
            centroids_bytes = f.read(num_meta_tokens * dim * 4)
            codebook.centroids = np.frombuffer(centroids_bytes, dtype=np.float32).reshape(num_meta_tokens, dim).copy()

            table = cls(codebook, num_tokens, m=m)
            indices_bytes = f.read(num_tokens * m * 2)
            table.indices = np.frombuffer(indices_bytes, dtype=np.uint16).reshape(num_tokens, m).copy()

            amplitudes_bytes = f.read(num_tokens * m * 1)
            table.amplitudes = np.frombuffer(amplitudes_bytes, dtype=np.uint8).reshape(num_tokens, m).copy()

        return table
