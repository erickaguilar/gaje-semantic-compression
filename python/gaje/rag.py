try:
    from .core._impl import NativeSemanticRAG as RustNativeSemanticRAG
except ImportError:
    from _impl import NativeSemanticRAG as RustNativeSemanticRAG


class NativeSemanticRAG:
    """
    🧬 Native Semantic RAG Engine
    Proporciona recuperación aumentada por generación ultrarrápida
    directamente en memoria compartida a través del kernel nativo de Rust.
    """

    def __init__(self):
        self._rag = RustNativeSemanticRAG()

    def add_document(self, text: str, embedding: list[float]):
        """Añade un documento y su vector de embedding al índice."""
        self._rag.add_document(text, embedding)

    def search(
        self, query_embedding: list[float], top_k: int = 3
    ) -> list[tuple[str, float]]:
        """Busca los top_k documentos más relevantes para el embedding de consulta."""
        return self._rag.search(query_embedding, top_k)

    def format_context(self, retrieved: list[tuple[str, float]]) -> str:
        """Formatea los documentos recuperados como contexto de prompt."""
        return self._rag.format_context(retrieved)

    def __len__(self):
        return self._rag.len()
