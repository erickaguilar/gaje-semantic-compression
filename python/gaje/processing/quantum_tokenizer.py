"""GAJE Helix — Quantum Genomic Tokenizer (Prototype).

Implements the mathematical isomorphism between 2-qubit Hilbert space:
{|00⟩=|A⟩, |01⟩=|C⟩, |10⟩=|G⟩, |11⟩=|T⟩}
and genomic 2-bit nucleotides, utilizing density matrices ρ and projective Born rule collapse.
"""

import math
import cmath
from typing import List, Tuple, Dict, Optional, Union
import numpy as np

# Bases estándar del espacio de Hilbert H^4 (2 qubits)
BASIS_A = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)  # |00⟩ = Adenina
BASIS_C = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.complex128)  # |01⟩ = Citosina
BASIS_G = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.complex128)  # |10⟩ = Guanina
BASIS_T = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.complex128)  # |11⟩ = Timina

GENOMIC_BASES = {"A": BASIS_A, "C": BASIS_C, "G": BASIS_G, "T": BASIS_T}
INDEX_TO_BASE = ["A", "C", "G", "T"]


class QuantumTokenState:
    """Representa el estado cuántico-genómico de un token en superposición o estado mixto."""

    def __init__(self, token_text: str, density_matrix: np.ndarray, pure_state: Optional[np.ndarray] = None):
        self.token_text = token_text
        self.rho = np.array(density_matrix, dtype=np.complex128)
        self.pure_state = pure_state

        # Asegurar propiedad hermítica y normalización de traza
        self._normalize_density_matrix()

    def _normalize_density_matrix(self):
        # ρ = (ρ + ρ†) / 2
        self.rho = 0.5 * (self.rho + np.conjugate(self.rho.T))
        tr = np.trace(self.rho).real
        if abs(tr) > 1e-12:
            self.rho = self.rho / tr
        else:
            self.rho = 0.25 * np.eye(4, dtype=np.complex128)

    @property
    def trace(self) -> float:
        """Retorna la traza de la matriz de densidad (debe ser 1.0)."""
        return float(np.trace(self.rho).real)

    @property
    def purity(self) -> float:
        """Calcula la pureza γ = Tr(ρ²). γ=1 para estado puro, γ=0.25 para estado máximamente mixto."""
        rho_sq = np.matmul(self.rho, self.rho)
        return float(np.trace(rho_sq).real)

    @property
    def von_neumann_entropy(self) -> float:
        """Calcula la entropía de von Neumann S(ρ) = -Tr(ρ log2 ρ) en bits."""
        eigenvalues = np.linalg.eigvalsh(self.rho)
        entropy = 0.0
        for val in eigenvalues:
            if val > 1e-12:
                entropy -= val * math.log2(val)
        return float(entropy)

    def collapse_with_context(self, context_vector: np.ndarray) -> Tuple[str, float]:
        """Proyecta el estado sobre el vector de contexto usando la regla de Born P = Tr(ρ |c⟩⟨c|)."""
        c_norm = context_vector / np.linalg.norm(context_vector)
        P_context = np.outer(c_norm, np.conjugate(c_norm))
        prob_overlap = float(np.trace(np.matmul(self.rho, P_context)).real)

        # Proyectar sobre cada una de las 4 bases genómicas canónicas
        probs = {}
        for base_name, base_vec in GENOMIC_BASES.items():
            P_base = np.outer(base_vec, np.conjugate(base_vec))
            # Medición compuesta: Tr(ρ · P_base) ponderada por contexto
            p = float(np.trace(np.matmul(self.rho, P_base)).real)
            # Factor de afinidad contextual
            p_contextual = p * float(np.abs(np.dot(np.conjugate(c_norm), base_vec)) ** 2)
            probs[base_name] = p_contextual

        total_p = sum(probs.values())
        if total_p > 1e-12:
            collapsed_base = max(probs.items(), key=lambda x: x[1])[0]
            confidence = probs[collapsed_base] / total_p
        else:
            collapsed_base = INDEX_TO_BASE[int(np.argmax(np.diag(self.rho).real))]
            confidence = float(np.diag(self.rho).real[INDEX_TO_BASE.index(collapsed_base)])

        return collapsed_base, confidence


class QuantumGenomicTokenizer:
    """
    🧬 QuantumGenomicTokenizer
    Tokenizador de nueva generación que transforma cadenas de texto en estados cuánticos |ψ⟩
    y matrices de densidad ρ, colapsándolos a secuencias de ADN de 2 bits mediante contexto semántico.
    """

    def __init__(self, default_ambiguity_smoothing: float = 0.05):
        self.smoothing = default_ambiguity_smoothing

    def encode_char_to_state(self, char: str) -> QuantumTokenState:
        """Mapea un carácter a un estado cuántico en superposición |ψ⟩ = α_A|A⟩ + α_C|C⟩ + α_G|G⟩ + α_T|T⟩."""
        code = ord(char)
        # Generar fases y amplitudes complejas a partir de la representación binaria y armónica
        theta_1 = (code % 360) * math.pi / 180.0
        theta_2 = ((code * 7) % 360) * math.pi / 180.0
        theta_3 = ((code * 13) % 360) * math.pi / 180.0
        theta_4 = ((code * 23) % 360) * math.pi / 180.0

        raw_amplitudes = np.array([
            cmath.rect(math.cos(theta_1) ** 2 + self.smoothing, theta_1),
            cmath.rect(math.sin(theta_2) ** 2 + self.smoothing, theta_2),
            cmath.rect(math.cos(theta_3) ** 2 + self.smoothing, theta_3),
            cmath.rect(math.sin(theta_4) ** 2 + self.smoothing, theta_4),
        ], dtype=np.complex128)

        norm = np.linalg.norm(raw_amplitudes)
        psi = raw_amplitudes / norm
        rho = np.outer(psi, np.conjugate(psi))

        return QuantumTokenState(token_text=char, density_matrix=rho, pure_state=psi)

    def encode_text_to_quantum_states(self, text: str) -> List[QuantumTokenState]:
        """Convierte una cadena de texto en una lista de estados cuánticos superpuestos."""
        return [self.encode_char_to_state(ch) for ch in text]

    def collapse_text_to_dna(self, text: str, context_text: Optional[str] = None) -> str:
        """
        Codifica un texto a estados cuánticos y los colapsa a una secuencia discreta de ADN (A, C, G, T)
        guiada por el contexto semántico o memoria episódica.
        """
        states = self.encode_text_to_quantum_states(text)

        if context_text:
            # Crear vector de contexto normalizado desde el texto de contexto
            ctx_hash = sum((idx + 1) * ord(c) for idx, c in enumerate(context_text))
            ctx_vec = np.array([
                math.cos(ctx_hash * 0.1),
                math.sin(ctx_hash * 0.2),
                math.cos(ctx_hash * 0.3),
                math.sin(ctx_hash * 0.4)
            ], dtype=np.complex128)
            ctx_vec = ctx_vec / np.linalg.norm(ctx_vec)
        else:
            # Contexto neutro uniforme
            ctx_vec = 0.5 * np.ones(4, dtype=np.complex128)

        dna_chars = []
        for st in states:
            base, _ = st.collapse_with_context(ctx_vec)
            dna_chars.append(base)

        return "".join(dna_chars)
