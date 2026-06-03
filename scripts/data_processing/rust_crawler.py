import requests
from bs4 import BeautifulSoup
import time
import os
from urllib.parse import urljoin

def rust_book_crawler(base_url, output_path="data/datasets/rust_documentation_full.txt"):
    """
    Crawler especializado para recorrer el índice de la documentación de Rust
    y descargar todos los capítulos.
    """
    print(f"🧬 Iniciando Operación 'Libro de Rust' en: {base_url}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        # 1. Obtener el índice principal
        response = requests.get(base_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 2. Identificar enlaces a capítulos
        # En la página principal de stable, buscamos enlaces relevantes
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            # Filtrar enlaces que parecen ser capítulos o sub-secciones
            if not href.startswith('http') and not href.startswith('#'):
                full_url = urljoin(base_url, href)
                if full_url not in links:
                    links.append(full_url)
        
        print(f"[*] Se encontraron {len(links)} posibles secciones para descargar.")
        
        total_chars = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for i, link in enumerate(links[:50]): # Limitamos a los primeros 50 para evitar bloqueos iniciales
                print(f"[{i+1}/{len(links)}] Descargando: {link}...")
                try:
                    res = requests.get(link, headers=headers, timeout=10)
                    if res.status_code == 200:
                        page_soup = BeautifulSoup(res.text, 'html.parser')
                        
                        # Extraer contenido significativo
                        # En mdbook (formato de Rust docs), el contenido está en <main>
                        main_content = page_soup.find('main')
                        if not main_content:
                            main_content = page_soup.find('article')
                        
                        if main_content:
                            content = []
                            for tag in main_content.find_all(['p', 'h1', 'h2', 'h3', 'code', 'li']):
                                text = tag.get_text().strip()
                                if len(text) > 15:
                                    content.append(text)
                            
                            clean_text = "\n\n".join(content)
                            f.write(f"\n--- SECCIÓN: {link} ---\n")
                            f.write(clean_text)
                            f.write("\n")
                            total_chars += len(clean_text)
                            
                        # Pequeña pausa para ser respetuosos
                        time.sleep(0.5)
                except Exception as e:
                    print(f"  ⚠️ Error en {link}: {e}")
        
        print(f"✅ Operación completada. Total extraído: {total_chars / 1024:.2f} KB")
        return True
        
    except Exception as e:
        print(f"❌ Error crítico en el crawler: {e}")
        return False

if __name__ == "__main__":
    # URL base de la documentación estable de Rust
    url = "https://doc.rust-lang.org/stable/"
    rust_book_crawler(url)
