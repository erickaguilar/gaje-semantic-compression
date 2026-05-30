import requests
from bs4 import BeautifulSoup


def simple_technical_scraper(base_url, output_path="data/datasets/scraped_docs.txt"):
    """
    Un scraper básico para extraer texto de documentación técnica.
    """
    print(f"🌐 Iniciando scraping en: {base_url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(base_url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # En documentación, el texto suele estar en <main>, <article> o divs específicos
        # Aquí buscamos párrafos y bloques de código
        content = []

        # Extraer texto de párrafos
        for p in soup.find_all(["p", "h1", "h2", "h3", "code"]):
            text = p.get_text().strip()
            if len(text) > 20:  # Filtrar ruido corto
                content.append(text)

        full_text = "\n\n".join(content)

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(f"\n--- FUENTE: {base_url} ---\n")
            f.write(full_text)
            f.write("\n")

        print(f"✅ Extraídos {len(full_text)} caracteres de {base_url}")
        return True

    except Exception as e:
        print(f"❌ Error al scrapear {base_url}: {e}")
        return False


if __name__ == "__main__":
    # Ejemplo con una sección del Rust Book (puedes cambiar la URL)
    url_to_scrape = "https://doc.rust-lang.org/book/ch04-01-what-is-ownership.html"
    simple_technical_scraper(url_to_scrape)
