import time

try:
    from ._impl import SessionBuffer
except ImportError:
    from _impl import SessionBuffer


class SessionMemory:
    """
    🧠 Capa de Sesión Toroidal (Ring Buffer)
    Gestiona la memoria intermedia de la sesión actual usando el núcleo nativo de Rust.
    Implementa el patrón de 'Memoria Semántica Recirculante'.
    """

    def __init__(self, capacity=1024, dim=1024, internal_buffer=None):
        """
        Inicializa la memoria de sesión.
        :param capacity: Número máximo de interacciones a recordar.
        :param dim: Dimensión de los vectores de fase (debe coincidir con el modelo).
        """
        if internal_buffer:
            self._buffer = internal_buffer
        else:
            self._buffer = SessionBuffer(capacity, dim)

    def push(self, text, phase_vector):
        """
        Inserta una interacción en el buffer circular.
        :param text: Texto de la interacción (ej. 'Usuario: Hola\nBot: Qué tal').
        :param phase_vector: Vector de estado complejo resultante de la inferencia.
        """
        # Aseguramos que el vector sea una lista para PyO3
        if not isinstance(phase_vector, list):
            phase_vector = list(phase_vector)

        self._buffer.push(text, phase_vector, int(time.time()))

    def retrieve(self, query_vector, top_k=3):
        """
        Recupera las interacciones más relevantes basándose en similitud de fase.
        :param query_vector: Vector de fase de la consulta actual.
        :param top_k: Número de fragmentos a recuperar.
        :return: Lista de strings con el contexto recuperado.
        """
        if not isinstance(query_vector, list):
            query_vector = list(query_vector)

        return self._buffer.retrieve_relevant(query_vector, top_k)

    def save(self, filepath):
        """
        Persiste el estado actual de la sesión en un archivo binario.
        Útil para alimentar el proceso de Direct Neural Ingestion (DNI) posterior.
        """
        self._buffer.dump_to_disk(filepath)

    @classmethod
    def load(cls, filepath):
        """
        Restaura una sesión desde un archivo binario.
        """
        internal = SessionBuffer.load_from_disk(filepath)
        return cls(internal_buffer=internal)

    def __len__(self):
        return len(self._buffer)
