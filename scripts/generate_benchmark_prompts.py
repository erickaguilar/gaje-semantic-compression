import os
import json

eval_dir = "data/eval"
os.makedirs(eval_dir, exist_ok=True)

prompts = {
    "version": "1.0.0",
    "description": "Banco de 25 Prompts Estandarizados de Evaluación GAJE Helix Benchmark Suite",
    "categories": [
        "factual_multilingual",
        "reasoning_math",
        "code_generation",
        "semantic_compression"
    ],
    "test_cases": [
        # === 1. FACTUALES MULTILINGÜES (10) ===
        {
            "id": "fact-es-01",
            "category": "factual_multilingual",
            "language": "es",
            "title": "Estructura del ADN",
            "prompt": "Explica de forma concisa qué es el ADN y cuáles son sus cuatro bases nitrogenadas.",
            "expected_keywords": ["adenina", "timina", "citosina", "guanina", "genético"],
            "max_target_tokens": 128
        },
        {
            "id": "fact-es-02",
            "category": "factual_multilingual",
            "language": "es",
            "title": "Sistema Solar",
            "prompt": "¿Cuáles son los planetas del sistema solar en orden desde el Sol?",
            "expected_keywords": ["Mercurio", "Venus", "Tierra", "Marte", "Júpiter", "Saturno", "Urano", "Neptuno"],
            "max_target_tokens": 96
        },
        {
            "id": "fact-es-03",
            "category": "factual_multilingual",
            "language": "es",
            "title": "Fotosíntesis",
            "prompt": "Describe el proceso de la fotosíntesis y su importancia para la vida en la Tierra.",
            "expected_keywords": ["luz", "dióxido de carbono", "oxígeno", "glucosa", "clorofila"],
            "max_target_tokens": 128
        },
        {
            "id": "fact-es-04",
            "category": "factual_multilingual",
            "language": "es",
            "title": "Teoría de la Relatividad",
            "prompt": "¿En qué consiste la equivalencia masa-energía formulada por Einstein?",
            "expected_keywords": ["E=mc²", "energía", "masa", "velocidad", "luz"],
            "max_target_tokens": 100
        },
        {
            "id": "fact-en-05",
            "category": "factual_multilingual",
            "language": "en",
            "title": "Quantum Superposition",
            "prompt": "Explain the concept of quantum superposition in two simple paragraphs.",
            "expected_keywords": ["state", "qubit", "probabilities", "measurement", "collapse"],
            "max_target_tokens": 128
        },
        {
            "id": "fact-en-06",
            "category": "factual_multilingual",
            "language": "en",
            "title": "Operating Systems Virtual Memory",
            "prompt": "What is memory-mapped file I/O (mmap) and why is it useful for low-latency systems?",
            "expected_keywords": ["mmap", "page", "kernel", "virtual memory", "zero-copy"],
            "max_target_tokens": 128
        },
        {
            "id": "fact-en-07",
            "category": "factual_multilingual",
            "language": "en",
            "title": "Cellular Respiration",
            "prompt": "Summarize the primary purpose of mitochondria in eukaryotic cells.",
            "expected_keywords": ["ATP", "energy", "cellular respiration", "mitochondria"],
            "max_target_tokens": 80
        },
        {
            "id": "fact-pt-08",
            "category": "factual_multilingual",
            "language": "pt",
            "title": "Exploração Espacial em Português",
            "prompt": "Escreva um breve resumo sobre a importância da exploração de Marte para a ciência moderna.",
            "expected_keywords": ["Marte", "ciência", "vida", "planeta", "exploração"],
            "max_target_tokens": 128
        },
        {
            "id": "fact-ja-09",
            "category": "factual_multilingual",
            "language": "ja",
            "title": "As quatro estações do Japão",
            "prompt": "日本の四季（春、夏、秋、冬）について、それぞれの特徴を簡潔に説明してください。",
            "expected_keywords": ["春", "夏", "秋", "冬", "桜", "紅葉"],
            "max_target_tokens": 160
        },
        {
            "id": "fact-es-10",
            "category": "factual_multilingual",
            "language": "es",
            "title": "Leyes de la Termodinámica",
            "prompt": "Enuncia y explica brevemente la primera y segunda ley de la termodinámica.",
            "expected_keywords": ["conservación", "energía", "entropía", "calor", "trabajo"],
            "max_target_tokens": 140
        },

        # === 2. RAZONAMIENTO LÓGICO Y MATEMÁTICO (5) ===
        {
            "id": "math-01",
            "category": "reasoning_math",
            "language": "es",
            "title": "Resolución de Ecuación Cuadrática",
            "prompt": "Resuelve paso a paso la ecuación cuadrática: 2x² - 8x + 6 = 0.",
            "expected_keywords": ["x = 1", "x = 3", "discriminante", "fórmula"],
            "max_target_tokens": 150
        },
        {
            "id": "math-02",
            "category": "reasoning_math",
            "language": "es",
            "title": "Problema de Probabilidad Básica",
            "prompt": "Si se lanzan dos dados estándar de 6 caras, ¿cuál es la probabilidad de que la suma sea igual a 7? Muestra los pares posibles.",
            "expected_keywords": ["6/36", "1/6", "(1,6)", "(2,5)", "(3,4)"],
            "max_target_tokens": 140
        },
        {
            "id": "math-03",
            "category": "reasoning_math",
            "language": "es",
            "title": "Lógica Deductiva (Silogismo)",
            "prompt": "Premisa 1: Todos los mamíferos tienen corazón.\nPremisa 2: Todos los delfines son mamíferos.\n¿Qué conclusión válida se deduce lógicamente?",
            "expected_keywords": ["delfines", "corazón", "conclusión"],
            "max_target_tokens": 80
        },
        {
            "id": "math-04",
            "category": "reasoning_math",
            "language": "es",
            "title": "Cálculo de Proporción y Descuento",
            "prompt": "Un producto cuesta $120 y tiene un descuento del 25%. Posteriormente se le aplica un impuesto del 10% sobre el precio con descuento. ¿Cuál es el precio final?",
            "expected_keywords": ["$90", "$99", "descuento", "precio final"],
            "max_target_tokens": 120
        },
        {
            "id": "math-05",
            "category": "reasoning_math",
            "language": "en",
            "title": "Algebraic Sequence Logic",
            "prompt": "Find the next two numbers in the sequence: 2, 6, 12, 20, 30, ... and explain the pattern.",
            "expected_keywords": ["42", "56", "n(n+1)", "difference", "pattern"],
            "max_target_tokens": 120
        },

        # === 3. GENERACIÓN DE CÓDIGO (5) ===
        {
            "id": "code-01",
            "category": "code_generation",
            "language": "es",
            "title": "Primer Carácter No Repetitivo en Python",
            "prompt": "Escribe una función en Python llamada `first_unique_char(s)` que reciba una cadena y devuelva el primer carácter que no se repite. Incluye docstring y un ejemplo de uso.",
            "expected_keywords": ["def first_unique_char", "return", "dict", "for", "char"],
            "max_target_tokens": 180
        },
        {
            "id": "code-02",
            "category": "code_generation",
            "language": "es",
            "title": "Búsqueda Binaria",
            "prompt": "Implementa en Python el algoritmo de búsqueda binaria iterativa sobre una lista ordenada de enteros.",
            "expected_keywords": ["def binary_search", "left", "right", "mid", "while"],
            "max_target_tokens": 160
        },
        {
            "id": "code-03",
            "category": "code_generation",
            "language": "es",
            "title": "Conteo de Frecuencia de Palabras",
            "prompt": "Crea una función en Python que tome un texto y devuelva un diccionario con la frecuencia de cada palabra ignorando mayúsculas y signos de puntuación.",
            "expected_keywords": ["lower()", "split()", "dict", "count", "return"],
            "max_target_tokens": 160
        },
        {
            "id": "code-04",
            "category": "code_generation",
            "language": "en",
            "title": "Fibonacci con Memoización",
            "prompt": "Write a Python function to compute the n-th Fibonacci number using memoization or lru_cache.",
            "expected_keywords": ["def fib", "memo", "lru_cache", "return", "n <= 1"],
            "max_target_tokens": 150
        },
        {
            "id": "code-05",
            "category": "code_generation",
            "language": "es",
            "title": "Verificación de Palíndromo",
            "prompt": "Escribe una función en Python de una sola línea o limpia que determine si una cadena de texto es un palíndromo.",
            "expected_keywords": ["def is_palindrome", "s == s[::-1]", "return"],
            "max_target_tokens": 100
        },

        # === 4. COMPRESIÓN SEMÁNTICA Y RESUMEN (5) ===
        {
            "id": "comp-01",
            "category": "semantic_compression",
            "language": "es",
            "title": "Resumen Ejecutivo de Texto Denso",
            "prompt": "Sintetiza en exactamente tres puntos clave la revolución de la computación distribuida en la nube y el edge computing.",
            "expected_keywords": ["latencia", "escalabilidad", "edge", "nube", "distribuido"],
            "max_target_tokens": 140
        },
        {
            "id": "comp-02",
            "category": "semantic_compression",
            "language": "es",
            "title": "Destilación de Concepto de Cuantización",
            "prompt": "Explica en menos de 50 palabras qué es la cuantización de pesos en redes neuronales (e.g., FP32 a 4-bit).",
            "expected_keywords": ["memoria", "bits", "pesos", "precisión", "compresión"],
            "max_target_tokens": 80
        },
        {
            "id": "comp-03",
            "category": "semantic_compression",
            "language": "es",
            "title": "Extracción de Entidades y Acción",
            "prompt": "Del siguiente enunciado: 'El 15 de marzo de 2026, AlphaCorp lanzó su satélite Geo-9 desde Cabo Cañaveral', extrae: Fecha, Empresa, Entidad lanzada y Ubicación.",
            "expected_keywords": ["15 de marzo de 2026", "AlphaCorp", "Geo-9", "Cabo Cañaveral"],
            "max_target_tokens": 100
        },
        {
            "id": "comp-04",
            "category": "semantic_compression",
            "language": "en",
            "title": "One-Sentence Semantic Summary",
            "prompt": "Condense the core idea of Shannon's Information Theory into a single concise sentence.",
            "expected_keywords": ["entropy", "information", "uncertainty", "communication", "channel"],
            "max_target_tokens": 60
        },
        {
            "id": "comp-05",
            "category": "semantic_compression",
            "language": "es",
            "title": "Analogía Genómica de IA",
            "prompt": "¿Por qué es útil comparar los pesos de un modelo de lenguaje comprimido con una secuencia de bases genómicas A, C, G, T?",
            "expected_keywords": ["información", "bases", "2 bits", "cuantización", "genómico"],
            "max_target_tokens": 130
        }
    ]
}

target_path = os.path.join(eval_dir, "benchmark_prompts.json")
with open(target_path, "w", encoding="utf-8") as f:
    json.dump(prompts, f, ensure_ascii=False, indent=2)

print(f"✅ Banco de {len(prompts['test_cases'])} prompts generado exitosamente en: {target_path}")
