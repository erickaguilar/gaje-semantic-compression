use tokenizers::Tokenizer;
use std::path::Path;
use std::error::Error;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg_attr(feature = "python", pyclass)]
#[derive(Clone)]
pub struct GajeTokenizer {
    inner: Tokenizer,
}

#[cfg_attr(feature = "python", pymethods)]
impl GajeTokenizer {
    #[cfg(feature = "python")]
    #[staticmethod]
    pub fn py_from_file(path: &str) -> PyResult<Self> {
        Self::from_file(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    #[cfg(feature = "python")]
    #[staticmethod]
    pub fn py_from_bytes(bytes: &[u8]) -> PyResult<Self> {
        Self::from_bytes(bytes).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }
}

impl GajeTokenizer {
    /// Carga el tokenizador desde un archivo JSON (formato HuggingFace)
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn Error>> {
        let inner = Tokenizer::from_file(path).map_err(|e| format!("Error cargando tokenizador: {}", e))?;
        Ok(Self { inner })
    }

    /// Carga el tokenizador desde bytes (JSON)
    pub fn from_bytes<B: AsRef<[u8]>>(bytes: B) -> Result<Self, Box<dyn Error>> {
        let inner = Tokenizer::from_bytes(bytes).map_err(|e| format!("Error cargando tokenizador desde bytes: {}", e))?;
        Ok(Self { inner })
    }

    /// Codifica un texto en una secuencia de IDs de tokens
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, Box<dyn Error>> {
        let encoding = self.inner.encode(text, add_special_tokens).map_err(|e| format!("Error en codificación: {}", e))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decodifica una secuencia de IDs de tokens en texto plano
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, Box<dyn Error>> {
        let decoded = self.inner.decode(ids, skip_special_tokens).map_err(|e| format!("Error en decodificación: {}", e))?;
        Ok(decoded)
    }

    /// Obtiene el tamaño del vocabulario
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Obtiene el ID de un token especial por nombre
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Serializa el tokenizador a una cadena JSON
    pub fn to_string(&self, pretty: bool) -> Result<String, Box<dyn Error>> {
        self.inner.to_string(pretty).map_err(|e| format!("Error serializando tokenizador: {}", e).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_tokenizer_loading() {
        // Este test asume que el archivo existe en el path estándar del proyecto
        let path = "models/core/tokenizer.json";
        if fs::metadata(path).is_ok() {
            let tokenizer = GajeTokenizer::from_file(path).unwrap();
            assert!(tokenizer.vocab_size() > 0);
            
            let text = "Hola mundo genómico";
            let ids = tokenizer.encode(text, true).unwrap();
            assert!(!ids.is_empty());
            
            let decoded = tokenizer.decode(&ids, true).unwrap();
            // El detokenizado podría tener ligeras variaciones de normalización (ej. espacios)
            assert!(!decoded.is_empty());
        }
    }
}
