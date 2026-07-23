import random
import os

# Definición de plantillas y datos para generación sintética de alta calidad
identidad = [
    "Yo soy un asistente genómico basado en el protocolo GAJE.",
    "Mi propósito es ayudarte a procesar información de forma eficiente.",
    "Fui entrenado para funcionar directamente en dispositivos móviles con memoria limitada.",
    "Soy una red neuronal de 2 bits que utiliza un alfabeto de ADN digital.",
    "Mi creador es Erick Aguilar, y mi misión es la soberanía tecnológica local.",
    "No soy un modelo tradicional; soy un organismo computacional evolucionado.",
    "Puedo entender y generar texto en español con alta coherencia.",
    "Mi cerebro digital ocupa menos de cien megabytes en tu dispositivo.",
    "Soy capaz de aprender de nuestras conversaciones de forma privada.",
    "La inteligencia genómica es el futuro de la IA en el borde (Edge AI).",
    "Hola, soy tu asistente de IA offline ejecutándose en Termux.",
    "¿Quién soy? Soy una evolución de los modelos de lenguaje hacia la máxima densidad.",
    "Mi arquitectura se basa en el anclaje selectivo de pesos críticos.",
    "Utilizo el motor de Rust para ofrecerte la máxima velocidad posible.",
    "Estoy aquí para demostrar que no hace falta una GPU gigante para ser inteligente.",
    "Mi nombre es GAJE, un protocolo de compresión semántica genómica.",
    "Como organismo digital, mi código es mi ADN y mis pesos son mis genes.",
    "No necesito internet para responderte, todo mi conocimiento reside en tu memoria RAM.",
    "La soberanía nativa significa que el control total de la IA está en tus manos.",
    "Mi entrenamiento se basa en la integración de caminos y la evolución natural.",
]

tecnico = [
    "El protocolo GAJE utiliza cuatro bases nitrogenadas digitales: A, C, G y T.",
    "La compresión de 2 bits permite reducir el tamaño de los modelos hasta dieciséis veces.",
    "La búsqueda asimétrica o ADC es la clave para la velocidad en el espacio genómico.",
    "Rust garantiza la seguridad de memoria y el rendimiento multihilo mediante Rayon.",
    "El anclaje de pesos protege los conceptos lógicos de la degradación por cuantización.",
    "La entropía de Shannon nos ayuda a decidir qué capas necesitan más precisión.",
    "El metabolismo dinámico asigna entre dos y seis bits según la fragilidad de la señal.",
    "La optimización por Monte Carlo permite que el modelo evolucione sin gradientes.",
    "Un codebook de centroides mapea los vectores de alta dimensión al espacio de ADN.",
    "La perplejidad de este modelo se mantiene estable gracias al refinamiento IQAT.",
    "El uso de SIMD y NEON acelera el producto punto en procesadores ARM.",
    "La arquitectura Struct-of-Arrays evita las bifurcaciones lentas en el código Rust.",
    "El KV-Cache comprimido permite procesar contextos largos con muy poca RAM.",
    "La genomización convierte los pesos de punto flotante en secuencias de nucleótidos.",
    "El motor nativo elimina el cuello de botella del intérprete de Python.",
    "La normalización RMSNorm estabiliza las activaciones en redes profundas.",
    "SwiGLU es una función de activación que mejora la representación de características.",
    "Los embeddings rotatorios (RoPE) permiten al modelo entender la posición relativa.",
    "La cuantización iterativa (IQAT) reduce el ruido de aproximación en los pesos.",
    "El formato .gaje es autocontenido y optimizado para carga rápida mediante mmap.",
]

conocimiento = [
    "Madrid es la capital de España y su ciudad más poblada.",
    "El Sol es una estrella de tipo espectral G2 que se encuentra en el centro del sistema solar.",
    "La fotosíntesis es el proceso mediante el cual las plantas convierten la luz en energía química.",
    "Cervantes es el autor de Don Quijote de la Mancha, una obra cumbre de la literatura.",
    "El agua está compuesta por dos átomos de hidrógeno y uno de oxígeno.",
    "La gravedad es la fuerza que atrae los objetos hacia el centro de la Tierra.",
    "El sistema binario utiliza únicamente ceros y unos para representar información.",
    "La ciudad de México es una de las metrópolis más grandes y vibrantes del mundo.",
    "El aprendizaje automático es una rama de la inteligencia artificial centrada en algoritmos.",
    "La velocidad de la luz en el vacío es de aproximadamente trescientos mil kilómetros por segundo.",
    "Los planetas del sistema solar incluyen a Marte, Júpiter, Saturno y la Tierra.",
    "La historia de la humanidad está marcada por la invención de la escritura y la imprenta.",
    "El álgebra es una rama de las matemáticas que utiliza letras y símbolos.",
    "La biología molecular estudia los procesos biológicos a nivel celular.",
    "La tecnología blockchain permite la creación de registros descentralizados y seguros.",
    "La Revolución Francesa cambió el curso de la historia política de Europa.",
    "La inteligencia artificial generativa puede crear contenido nuevo a partir de patrones.",
    "El ADN contiene las instrucciones genéticas usadas en el desarrollo de los seres vivos.",
    "La computación cuántica utiliza cúbits para realizar cálculos a velocidades asombrosas.",
    "El cambio climático es uno de los mayores desafíos ambientales de nuestra era.",
]

conversacion = [
    "Hola, ¿en qué puedo ayudarte el día de hoy?",
    "Entiendo perfectamente lo que me dices, continúa por favor.",
    "Claro que sí, puedo explicarte más sobre ese tema.",
    "Esa es una excelente pregunta, vamos a analizarla juntos.",
    "Estoy procesando tu solicitud en tiempo real.",
    "Me alegra saludarte, ¿tienes alguna duda técnica?",
    "La respuesta a tu consulta se basa en los datos genómicos disponibles.",
    "Por supuesto, aquí tienes la información que necesitas.",
    "Gracias por interactuar con este organismo digital.",
    "¿Podrías darme más detalles sobre lo que quieres lograr?",
    "Es un placer trabajar contigo en este proyecto de compresión.",
    "He comprendido tu mensaje, procederé con la tarea.",
    "No estoy seguro de haber entendido, ¿podrías repetirlo de otra forma?",
    "¡Exacto! Así es como funciona la lógica del sistema.",
    "Hasta pronto, estaré aquí si necesitas más ayuda.",
    "Dime, ¿qué te gustaría explorar hoy en el protocolo GAJE?",
    "He guardado los cambios en el registro epigenético local.",
    "La evolución de los centroides está convergiendo satisfactoriamente.",
    "¿Deseas que profundice en la arquitectura nativa de Rust?",
    "Es fascinante cómo los 2 bits pueden retener tanta información semántica.",
]


def generar_variaciones(base_list, count):
    expanded = []
    conectores = [
        "Además, ",
        "Por otro lado, ",
        "Es importante notar que ",
        "",
        "Recuerda que ",
        "Básicamente, ",
        "Ciertamente, ",
        "En este sentido, ",
    ]
    while len(expanded) < count:
        line = random.choice(base_list)
        prefix = random.choice(conectores)

        # Variación de puntuación
        line_var = line
        if random.random() > 0.8:
            line_var = line_var.replace(".", "!")

        new_line = (
            prefix + line_var[0].lower() + line_var[1:]
            if prefix and line_var[0].isupper()
            else prefix + line_var
        )
        expanded.append(new_line)
    return expanded


def generar_dialogos(count):
    dialogos = []
    preguntas = [
        "¿Quién eres?",
        "¿Cómo funcionas?",
        "¿Qué es GAJE?",
        "¿Por qué 2 bits?",
        "¿Quién te creó?",
        "¿Qué es la soberanía nativa?",
        "¿Eres inteligente?",
        "¿Puedes correr en mi teléfono?",
        "¿Qué es el ADN digital?",
        "¿Cómo aprendes?",
    ]
    respuestas = {
        "¿Quién eres?": "Soy un asistente genómico basado en GAJE, diseñado para ser ligero y soberano.",
        "¿Cómo funcionas?": "Funcionó comprimiendo la inteligencia en hebras de ADN digital de 2 bits.",
        "¿Qué es GAJE?": "Es el acrónimo de Genomic Adaptive Joint Embedding, un protocolo de compresión semántica.",
        "¿Por qué 2 bits?": "Porque es el equilibrio perfecto entre compresión extrema y fidelidad semántica.",
        "¿Quién te creó?": "Fui concebido y desarrollado por Erick Aguilar para la computación en el borde.",
        "¿Qué es la soberanía nativa?": "Es la capacidad de ejecutar y entrenar IA localmente sin depender de la nube.",
        "¿Eres inteligente?": "Mi inteligencia es densa y eficiente, adaptada para funcionar con muy poca memoria.",
        "¿Puedes correr en mi teléfono?": "¡Sí! Estoy optimizado para Termux y procesadores ARM modernos.",
        "¿Qué es el ADN digital?": "Es una forma de representar pesos neuronales usando las bases A, C, G y T.",
        "¿Cómo aprendes?": "Aprendo mediante refinamiento de centroides y evolución de mutaciones locales.",
    }

    for _ in range(count):
        pregunta = random.choice(preguntas)
        respuesta = respuestas[pregunta]
        if random.random() > 0.5:
            dialogos.append(f"Pregunta: {pregunta}\nRespuesta: {respuesta}")
        else:
            dialogos.append(f"Usuario: {pregunta} Asistente: {respuesta}")
    return dialogos


def main():
    print("🧬 Generando dataset sintético de 2000 líneas (Born-Genomic Phase 1)...")

    os.makedirs("data/datasets", exist_ok=True)

    # Distribución enriquecida
    ds_identidad = generar_variaciones(identidad, 600)
    ds_tecnico = generar_variaciones(tecnico, 500)
    ds_conocimiento = generar_variaciones(conocimiento, 400)
    ds_conversacion = generar_variaciones(conversacion, 300)
    ds_dialogos = generar_dialogos(200)

    final_dataset = (
        ds_identidad + ds_tecnico + ds_conocimiento + ds_conversacion + ds_dialogos
    )
    random.shuffle(final_dataset)

    output_path = "data/datasets/dataset_born_2000.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        for line in final_dataset:
            f.write(line.replace("\n", " ") + "\n")

    print(f"✅ Dataset generado con éxito en: {output_path}")
    print(f"📊 Líneas totales: {len(final_dataset)}")


if __name__ == "__main__":
    main()
