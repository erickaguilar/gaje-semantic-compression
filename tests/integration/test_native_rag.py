import numpy as np
from gaje.rag import NativeSemanticRAG


def test_native_rag():
    print("🧬 GAJE Phase 4 Test: Native Semantic RAG Kernel Verification")
    print("=" * 60)

    rag = NativeSemanticRAG()

    # Documentos de prueba con embeddings sintéticos ordenados
    doc1 = "El motor GAJE utiliza compresión semántica genómica a 2 bits."
    doc2 = "La velocidad de ingestión neuronal directa (DNI) utiliza mutación acelerada por GPU/CPU."
    doc3 = "La perplejidad del modelo Silver Adult disminuye con inhibición lateral K-WTA."

    # Simulación de embeddings de dimensión 128
    vec1 = np.random.randn(128).tolist()
    vec2 = np.random.randn(128).tolist()
    vec3 = np.random.randn(128).tolist()

    rag.add_document(doc1, vec1)
    rag.add_document(doc2, vec2)
    rag.add_document(doc3, vec3)

    assert len(rag) == 3, f"Expected 3 documents, got {len(rag)}"
    print(f"[*] {len(rag)} documentos indexados en el motor RAG Nativo.")

    # Búsqueda con el embedding idéntico al documento 1 (similitud esperada ~1.0)
    results = rag.search(vec1, top_k=2)
    print(f"[*] Resultados de búsqueda para consulta 1:")
    for text, score in results:
        print(f"    - [{score:.4f}] {text}")

    assert results[0][0] == doc1, "Top 1 result should match doc1 exactly"
    assert results[0][1] >= 0.999, f"Similarity score should be ~1.0, got {results[0][1]}"

    formatted = rag.format_context(results)
    print("\n[*] Contexto Formateado para Prompt:")
    print(formatted)

    print("\n" + "=" * 60)
    print("✅ CERTIFICACIÓN FASE 4 (NATIVE RAG): PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_native_rag()
