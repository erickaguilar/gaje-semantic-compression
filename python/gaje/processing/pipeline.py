import os
import numpy as np
import gguf
import time
from transformers import AutoTokenizer
from gaje.core import _impl as dna_semantic_compression
from gaje.nn.stabilized import GenomicLayer, GenomicTransformerBlock, GenomicLLM

# Este archivo se mantiene por compatibilidad con benchmarks previos
# pero ahora redirige al motor estabilizado v0.6.0 para evitar errores de Q8

__all__ = ["GenomicLayer", "GenomicTransformerBlock", "GenomicLLM"]
