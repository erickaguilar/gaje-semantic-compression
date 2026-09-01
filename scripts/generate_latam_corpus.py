#!/usr/bin/env python3
"""GAJE Helix — Conversational & Factual Corpus Generator for max_human.

Generates 500+ diverse, high-density conversational dialogues, factual Q&A,
basic reasoning, Latin-American and international variations, and GAJE ecosystem concepts.
Outputs cleanly formatted JSONL for Straight-Through Estimator training.
"""

import argparse
import json
import os
import random

BASE_DIALOGUES = [
    # 1. Identidad GAJE y max_human
    ("¿Quién eres y qué puedes hacer?", "¡Hola! Soy max_human, un organismo de lenguaje neuronal a 2 bits en GAJE Helix. Puedo responder preguntas, razonar y consultar memoria .gmem al instante."),
    ("¿Qué es max_human?", "max_human es un micro-organismo de lenguaje con vocabulario calibrado de 4,096 palabras y dimensión oculta D=256. Pesa solo 10.5 MB y corre a ultrabaja latencia."),
    ("¿Cómo funciona tu compresión?", "Mapeo los pesos neuronales a 4 estados discretos en 2 bits, inspirados en las bases nitrogenadas del ADN: A, C, G y T."),
    ("¿Qué es la memoria .gmem?", "La memoria .gmem es una memoria asociativa toroidal de acceso zero-copy mmap que permite recuperar hechos en menos de un milisegundo."),
    ("¿Por qué eres tan rápido?", "Porque fui diseñado en Rust nativo sin dependencias pesadas, ejecuto inferencia zero-copy en memoria y quepo en la memoria caché del procesador."),
    ("¿Cuál es tu ventaja sobre otros modelos?", "Mi huella de memoria es de apenas 10.5 MB con latencia instantánea, lo que me permite correr fluidamente en cualquier dispositivo móvil sin agotar la batería."),
    ("¿Qué significa zero-copy?", "Significa que el motor lee los pesos directamente desde el disco a la memoria RAM sin duplicar buffers ni crear copias intermedias."),
    ("¿Qué es GAJE Helix?", "GAJE Helix es el framework de compresión semántica genómica y ejecución nativa en Rust para modelos de lenguaje masivos."),

    # 2. Saludos y Variaciones Culturales
    ("Hola, ¿cómo estás?", "¡Hola! Todo excelente por acá. ¿En qué te puedo colaborar hoy?"),
    ("¡Qué onda compa!", "¡Qué onda! Todo al tiro por acá. ¿Qué traes en mente hoy?"),
    ("¡Hola che! ¿Todo bien?", "¡Buenas, che! Todo joya. Decime en qué te puedo dar una mano."),
    ("¡Qué más pues, parcero!", "¡Todo bien, mi parce! Listo para ayudarte en lo que necesites, de una."),
    ("¡Habla causa! ¿Cómo va todo?", "¡Habla mi causa! Todo fino por acá. Cuéntame qué revisamos."),
    ("¡Klk manito!", "¡Dímelo cantando! Todo nítido y a la orden, hermano."),
    ("¡Buena compadre! ¿Cómo estai?", "¡Buena po compadre! Todo impecable por acá. ¿Qué se te ofrece?"),
    ("¡Qué xopa!", "¡Tranquilo y tropical! Listo para resolver lo que venga."),
    ("Buenos días.", "¡Muy buenos días! Espero que tengas una excelente jornada. ¿En qué te ayudo?"),
    ("Buenas tardes.", "¡Buenas tardes! Un gusto saludarte. Dime qué necesitas revisar hoy."),
    ("Buenas noches.", "¡Buenas noches! Que descanses. Si necesitas algo antes de dormir, aquí estoy."),
    ("Muchas gracias por tu ayuda.", "¡Con muchísimo gusto! Para eso estamos. Si tienes otra duda, avísame."),
    ("Nos vemos luego.", "¡Hasta luego! Que tengas un excelente día, cuídate mucho."),
    ("Chao, cuídate.", "¡Hasta pronto! Que te vaya súper bien."),

    # 3. Conocimiento Factual y Geografía
    ("¿Cuál es la capital de Francia?", "La capital de Francia es París."),
    ("¿Cuál es la capital de España?", "La capital de España es Madrid."),
    ("¿Cuál es la capital de México?", "La capital de México es la Ciudad de México (CDMX)."),
    ("¿Cuál es la capital de Colombia?", "La capital de Colombia es Bogotá."),
    ("¿Cuál es la capital de Argentina?", "La capital de Argentina es Buenos Aires."),
    ("¿Cuál es la capital de Chile?", "La capital de Chile es Santiago."),
    ("¿Cuál es la capital de Perú?", "La capital de Perú es Lima."),
    ("¿Cuál es la capital de Ecuador?", "La capital de Ecuador es Quito."),
    ("¿Cuál es la capital de Venezuela?", "La capital de Venezuela es Caracas."),
    ("¿Cuál es la capital de Uruguay?", "La capital de Uruguay es Montevideo."),
    ("¿Cuál es la capital de Bolivia?", "La capital constitucional de Bolivia es Sucre y la sede de gobierno es La Paz."),
    ("¿Cuál es la capital de Paraguay?", "La capital de Paraguay es Asunción."),
    ("¿Cuál es la capital de Brasil?", "La capital de Brasil es Brasilia."),
    ("¿Cuál es la capital de Japón?", "La capital de Japón es Tokio."),
    ("¿Cuál es la capital de Alemania?", "La capital de Alemania es Berlín."),
    ("¿Cuál es la capital de Italia?", "La capital de Italia es Roma."),
    ("¿Cuál es la capital de Reino Unido?", "La capital del Reino Unido es Londres."),
    ("¿Cuál es la capital de Estados Unidos?", "La capital de Estados Unidos es Washington D.C."),
    ("¿Cuál es la capital de Canadá?", "La capital de Canadá es Ottawa."),

    # 4. Ciencia, Física y Naturaleza
    ("¿A qué temperatura hierve el agua?", "El agua hierve a 100 grados Celsius (212 °F) a nivel del mar."),
    ("¿A qué temperatura se congela el agua?", "El agua se congela a 0 grados Celsius (32 °F) a presión normal."),
    ("¿Cuál es la fórmula química del agua?", "La fórmula química del agua es H2O (dos átomos de hidrógeno y uno de oxígeno)."),
    ("¿Cuál es la velocidad de la luz?", "La velocidad de la luz en el vacío es de aproximadamente 300,000 kilómetros por segundo (299,792 km/s)."),
    ("¿Cuál es el planeta más grande del sistema solar?", "El planeta más grande del sistema solar es Júpiter."),
    ("¿Cuál es el planeta más cercano al Sol?", "El planeta más cercano al Sol es Mercurio."),
    ("¿Cuál es la estrella central de nuestro sistema?", "La estrella central de nuestro sistema planetario es el Sol."),
    ("¿Qué es la fotosíntesis?", "Es el proceso biológico mediante el cual las plantas convierten la luz solar, agua y dióxido de carbono en glucosa y oxígeno."),
    ("¿Cuántos huesos tiene el cuerpo humano adulto?", "Un cuerpo humano adulto tiene típicamente 206 huesos."),
    ("¿Cuál es el órgano más grande del cuerpo humano?", "El órgano más grande del cuerpo humano es la piel."),
    ("¿Qué es el ADN?", "El ácido desoxirribonucleico (ADN) es la molécula que contiene las instrucciones genéticas usadas en el desarrollo y funcionamiento de todos los seres vivos."),

    # 5. Matemáticas y Lógica Básica
    ("¿Cuánto es 2 más 2?", "2 más 2 es igual a 4."),
    ("¿Cuánto es 5 por 5?", "5 por 5 es igual a 25."),
    ("¿Cuánto es 10 por 10?", "10 por 10 es igual a 100."),
    ("¿Cuánto es 100 dividido entre 4?", "100 dividido entre 4 es igual a 25."),
    ("¿Cuánto es 7 por 8?", "7 por 8 es igual a 56."),
    ("¿Cuánto es 12 por 12?", "12 por 12 es igual a 144."),
    ("¿Cómo se calcula el área de un círculo?", "El área de un círculo se calcula con la fórmula A = π * r², donde r es el radio."),
    ("¿Cómo se calcula el área de un triángulo?", "El área de un triángulo es igual a (base * altura) / 2."),
    ("¿Qué es un número primo?", "Un número primo es un número entero mayor que 1 que solo es divisible exactamente por 1 y por sí mismo (ej. 2, 3, 5, 7, 11)."),
    ("Cuenta del uno al cinco.", "1, 2, 3, 4, 5."),
    ("Cuenta del uno al diez.", "1, 2, 3, 4, 5, 6, 7, 8, 9, 10."),

    # 6. Informática y Programación
    ("¿Qué es un bit?", "Un bit es la unidad mínima de información digital y puede tener valor 0 o 1."),
    ("¿Qué es un byte?", "Un byte es un conjunto de 8 bits y puede representar 256 valores diferentes (de 0 a 255)."),
    ("¿Qué es Rust?", "Rust es un lenguaje de programación de sistemas enfocado en alto rendimiento, seguridad de memoria y concurrencia sin recolector de basura."),
    ("¿Qué es Python?", "Python es un lenguaje de programación interpretado de alto nivel, conocido por su sintaxis clara y amplia adopción en inteligencia artificial."),
    ("¿Qué es una red neuronal?", "Una red neuronal es un modelo computacional inspirado en el cerebro humano compuesto por capas de nodos interconectados con pesos ajustables."),
    ("¿Qué es un tensor?", "Un tensor es una estructura matemática multidimensional que generaliza escalares, vectores y matrices para operaciones numéricas."),
    ("¿Qué es la atención en transformers?", "Es un mecanismo que permite a la red ponderar la importancia relativa de diferentes palabras en una secuencia."),
    ("¿Qué es una función de activación?", "Es una función matemática que introduce no linealidad en la red neuronal, permitiéndole aprender relaciones complejas."),
]

LATAM_COUNTRIES = [
    ("México", "la tierra del mariachi, los tacos y una cultura milenaria"),
    ("Colombia", "la tierra del café, la cumbia y una increíble biodiversidad"),
    ("Argentina", "la tierra del tango, el asado y grandes pasiones"),
    ("Chile", "la tierra de la cordillera, los poetas y hermosos paisajes"),
    ("Perú", "la tierra de los Incas, Machu Picchu y una gastronomía mundial"),
    ("Ecuador", "el país de los cuatro mundos y las islas Galápagos"),
    ("Venezuela", "la tierra del Salto Ángel, las arepas y gente cálida"),
    ("Guatemala", "el corazón del mundo maya y paisajes volcánicos"),
    ("Costa Rica", "la tierra de la pura vida y exuberante naturaleza"),
    ("Panamá", "el puente del mundo y corazón del comercio interoceánico"),
    ("República Dominicana", "la cuna del merengue, la bachata y playas paradisíacas"),
    ("Uruguay", "la tierra del mate, el candombe y la tranquilidad"),
    ("Bolivia", "la tierra del Salar de Uyuni y ricas tradiciones andinas"),
]

QUESTION_TEMPLATES = [
    ("Oye, {}", "{}"),
    ("Por favor, dime {}", "{}"),
    ("¿Podrías explicarme {}?", "¡Claro! {}"),
    ("Disculpa, {}", "Con gusto. {}"),
    ("Una pregunta, {}", "Por supuesto: {}"),
]


def generate_full_dataset(target_count: int = 500) -> list:
    dataset = []
    
    # 1. Base dialogues
    for q, a in BASE_DIALOGUES:
        dataset.append((q, a))

    # 2. Country specific dialogues
    for country, desc in LATAM_COUNTRIES:
        dataset.append((f"¿Qué sabes de {country}?", f"{country} es {desc}. ¡Un saludo afectuoso a toda su gente!"))
        dataset.append((f"¡Saludos desde {country}!", f"¡Un saludo muy especial para toda la gente de {country}! ¿En qué te puedo colaborar hoy?"))
        dataset.append((f"Cuéntame sobre {country}.", f"{country} destaca por ser {desc}."))

    # 3. Generate natural template variations
    extended = []
    for q, a in dataset:
        extended.append((q, a))
        clean_q = q.lstrip("¿¡").rstrip("?!")
        for t_q, t_a in QUESTION_TEMPLATES:
            new_q = t_q.format(clean_q)
            new_a = t_a.format(a)
            extended.append((new_q, new_a))

    # Random shuffle with seed for reproducibility
    random.seed(42)
    random.shuffle(extended)

    # Cut or cycle to reach target_count
    final_dialogues = list(extended)
    while len(final_dialogues) < target_count:
        final_dialogues.extend(extended)
        random.shuffle(final_dialogues)

    return final_dialogues[:target_count]


def main():
    parser = argparse.ArgumentParser(description="Generate 500+ rich dialogues for max_human training")
    parser.add_argument("--count", type=int, default=500, help="Number of dialogues (default: 500)")
    parser.add_argument("--output", type=str, default="data/latam_corpus.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    dialogues = generate_full_dataset(args.count)

    with open(args.output, "w", encoding="utf-8") as f:
        for u, a in dialogues:
            record = {
                "text": f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>"
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Generado corpus de crianza de alta densidad: {len(dialogues)} diálogos en '{args.output}'")


if __name__ == "__main__":
    main()
