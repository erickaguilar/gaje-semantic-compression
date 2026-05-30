try:
    from ._impl import DNIEngine as RustDNIEngine
except ImportError:
    from _impl import DNIEngine as RustDNIEngine

class DNIEngine:
    """
    🧬 Direct Neural Ingestion (DNI) Engine
    Permite la inyección directa de conocimiento en los pesos del modelo
    mediante evolución genética acelerada.
    """
    def __init__(self, model, tokenizer, council=None, intensity=0.01):
        """
        Inicializa el motor DNI.
        :param model: Instancia de GenomicLLM (motor de Rust).
        :param tokenizer: Instancia de GajeTokenizer (motor de Rust).
        :param council: Opcional, consejo de maestros para destilación.
        :param intensity: Ratio de mutación inicial (0.0 a 1.0).
        """
        # Necesitamos extraer el objeto rust_llm de la clase GenomicLLM de Python
        # si es que se está pasando el wrapper de Python.
        rust_model = getattr(model, "rust_llm", model)
        
        # Lo mismo para el tokenizer si es un objeto complejo
        # Pero GajeTokenizer suele ser un objeto nativo directo en algunos scripts.
        # Por ahora asumimos que se pasan los objetos nativos o compatibles.
        self._engine = RustDNIEngine(rust_model, tokenizer, council, intensity)

    def ingest(self, text, generations=100, pop_size=16):
        """
        Ejecuta el proceso de inyección de un texto en los pesos del modelo.
        :param text: El fragmento de información a ingerir.
        :param generations: Número de ciclos evolutivos.
        :param pop_size: Tamaño de la población de mutantes por generación.
        :return: Fitness final alcanzado (0.0 a 1.0).
        """
        return self._engine.ingest_text(text, generations, pop_size)

    @property
    def model(self):
        """Devuelve el modelo con los pesos actualizados."""
        return self._engine.model
