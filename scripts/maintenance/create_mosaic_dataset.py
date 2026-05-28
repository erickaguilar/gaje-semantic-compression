import os
import random
import re

def is_spanish(text):
    # Heurística simple para detectar español
    spanish_words = r"\b(que|el|en|de|un|es|por|para|con|las|los|una|del|al|lo|se|su|al|un|esta|este)\b"
    matches = re.findall(spanish_words, text.lower())
    return len(matches) >= 3

def create_mosaic():
    print("🧬 Creando Mosaic Dataset (Equilibrio Semántico)...")
    
    datasets_dir = "data/datasets"
    sources = {
        "cultural": os.path.join(datasets_dir, "ai_culture_multilingual.txt"),
        "technical": os.path.join(datasets_dir, "consolidated_silver_dataset.txt"),
        "chat": os.path.join(datasets_dir, "synthetic_logic_robust.txt")
    }
    
    # 1. Extraer Cultural (40%)
    print("[*] Extrayendo cultura general en español...")
    cultural_lines = []
    if os.path.exists(sources["cultural"]):
        with open(sources["cultural"], "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if len(line) > 50 and is_spanish(line):
                    cultural_lines.append(line)
                    if len(cultural_lines) >= 10000: break # Límite para esta versión
    
    # 2. Extraer Técnico (30%)
    print("[*] Extrayendo conocimiento técnico...")
    technical_lines = []
    if os.path.exists(sources["technical"]):
        with open(sources["technical"], "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if len(line) > 20:
                    technical_lines.append(line)
    
    # 3. Extraer Chat/Lógica (30%)
    print("[*] Extrayendo interacción y lógica...")
    chat_lines = []
    if os.path.exists(sources["chat"]):
        with open(sources["chat"], "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if len(line) > 20:
                    chat_lines.append(line)

    print(f"📊 Fuentes cargadas: Cultural({len(cultural_lines)}), Técnico({len(technical_lines)}), Chat({len(chat_lines)})")

    # Ajustar para mantener el balance 40/30/30
    # Queremos un dataset de aproximadamente 50,000 líneas para esta prueba de micro-organismo
    target_total = 50000
    n_cultural = int(target_total * 0.4)
    n_technical = int(target_total * 0.3)
    n_chat = int(target_total * 0.3)

    final_lines = []
    
    # Muestreo con reemplazo si es necesario para llegar al target
    if cultural_lines:
        final_lines.extend(random.choices(cultural_lines, k=n_cultural))
    if technical_lines:
        final_lines.extend(random.choices(technical_lines, k=n_technical))
    if chat_lines:
        final_lines.extend(random.choices(chat_lines, k=n_chat))

    random.shuffle(final_lines)
    
    output_path = os.path.join(datasets_dir, "mosaic_dataset.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        for line in final_lines:
            f.write(line + "\n")
            
    print(f"✅ Mosaic Dataset creado en: {output_path}")
    print(f"📈 Tamaño total: {len(final_lines)} líneas.")

if __name__ == "__main__":
    create_mosaic()
