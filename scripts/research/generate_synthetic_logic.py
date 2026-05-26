import os
import random

def generate_synthetic_logic_es(count=5000):
    """
    Genera patrones lógicos y de instrucción sintéticos para fortalecer la gramática.
    """
    templates = [
        "Pregunta: ¿Cómo funciona {tema}?\nRespuesta: El funcionamiento de {tema} se basa en {mecanismo}, lo que permite {beneficio}.",
        "Instrucción: Explica la importancia de {tema} en Rust.\nRespuesta: En Rust, {tema} es fundamental para garantizar {beneficio} mediante {mecanismo}.",
        "Análisis Técnico: El componente {tema} presenta una estructura de {mecanismo}. Esto optimiza el rendimiento en un {porcentaje}% en sistemas distribuidos.",
        "Error Común: Muchos desarrolladores confunden {tema} con {alternativa}. La diferencia clave radica en {mecanismo}.",
        "Código Ejemplo (Rust):\nfn {funcion}_test() {{\n    let {variable} = {mecanismo};\n    println!(\"El valor de {tema} es: {{:?}}\", {variable});\n}}",
    ]
    
    temas = ["la gestión de memoria", "la propiedad (ownership)", "el préstamo (borrowing)", "los rasgos (traits)", "la concurrencia nativa", "la compresión genómica", "los tensores de 2 bits", "la perplejidad diferencial"]
    mecanismos = ["el verificador de préstamos", "el conteo de referencias", "la seguridad en tiempo de compilación", "la cuantización de centroides", "la matriz de adyacencia relacional", "el paralelismo con Rayon", "la optimización SIMD"]
    beneficios = ["evitar fugas de memoria", "garantizar la seguridad de hilos", "maximizar la eficiencia en ARM", "reducir el semantic drift", "lograr una inferencia zero-gil"]
    alternativas = ["punteros crudos", "recolección de basura", "copia profunda", "memoria compartida mutable", "cuantización escalar"]
    funciones = ["check_memory", "optimize_genome", "distill_knowledge", "validate_needle", "compute_adc"]
    variables = ["genome_str", "tensor_bits", "anchor_weight", "ppl_score", "bias_factor"]
    
    synthetic_data = []
    for _ in range(count):
        t = random.choice(templates)
        data = t.format(
            tema=random.choice(temas),
            mecanismo=random.choice(mecanismos),
            beneficio=random.choice(beneficios),
            alternativa=random.choice(alternativas),
            funcion=random.choice(funciones),
            variable=random.choice(variables),
            porcentaje=random.randint(20, 95)
        )
        synthetic_data.append(data)
    
    return "\n\n".join(synthetic_data)

if __name__ == "__main__":
    print("🧠 Generando 5000 muestras de Lógica Sintética (Español/Rust)...")
    data = generate_synthetic_logic_es(5000)
    
    output_path = "data/datasets/synthetic_logic_robust.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(data)
    
    print(f"✅ Lógica sintética guardada en: {output_path} ({len(data.encode('utf-8')) / 1024:.2f} KB)")
