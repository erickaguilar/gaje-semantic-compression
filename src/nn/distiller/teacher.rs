// =============================================================================
// teacher — Teacher: maestro GGUF con mapeo de vocabulario
// =============================================================================
#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::core::tokenizer::GajeTokenizer;
#[cfg(feature = "native")]
use crate::io::loader::GGUFLoader;
use crate::nn::llm::GenomicLLM;

/// Representa un maestro en el Consejo de Profesores.
#[cfg_attr(feature = "python", pyclass)]
#[derive(Clone)]
pub struct Teacher {
    pub name: String,
    pub model: GenomicLLM,
    pub tokenizer: GajeTokenizer,
    pub vocab_mapping: Vec<Option<usize>>, // teacher_token_id -> student_token_id
    pub is_identity_vocab: bool,
}

#[cfg_attr(feature = "python", pymethods)]
impl Teacher {
    /// Crea un nuevo maestro cargando un modelo GGUF y su tokenizador.
    #[cfg(all(feature = "python", feature = "native"))]
    #[new]
    pub fn py_new(
        name: String,
        model_path: &str,
        tokenizer_path: &str,
        student_tokenizer: &GajeTokenizer,
    ) -> PyResult<Self> {
        Self::new(name, model_path, tokenizer_path, student_tokenizer)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }
}

impl Teacher {
    /// Crea un nuevo maestro cargando un modelo GGUF y su tokenizador.
    /// El mapeo de vocabulario se pre-calcula comparando tokens decodificados.
    #[cfg(feature = "native")]
    pub fn new(
        name: String,
        model_path: &str,
        tokenizer_path: &str,
        student_tokenizer: &GajeTokenizer,
    ) -> Result<Self, String> {
        println!("[*] Cargando Maestro '{}' desde {}...", name, model_path);
        let loader = GGUFLoader::new(model_path).map_err(|e| e.to_string())?;
        let config = loader.infer_config().map_err(|e| e.to_string())?;

        // Cargamos el maestro como un modelo genómico para ejecución nativa en Rust.
        let model = loader
            .load_genomic_llm(config, -1.0)
            .map_err(|e| e.to_string())?;

        let tokenizer = GajeTokenizer::from_file(tokenizer_path).map_err(|e| e.to_string())?;

        let vocab_size = tokenizer.vocab_size();
        let mut vocab_mapping = vec![None; vocab_size];

        println!(
            "[*] Pre-calculando mapeo de vocabulario para '{}' (Vocab: {})...",
            name, vocab_size
        );

        // Optimización: Si los tokenizadores son idénticos (basado en el tamaño del vocabulario
        // y una prueba rápida de los primeros 100 tokens), podemos usar un mapeo de identidad.
        let mut is_identity = vocab_size == student_tokenizer.vocab_size();
        if is_identity {
            for i in 0..100.min(vocab_size) {
                if let Ok(t1) = tokenizer.decode(&[i as u32], true) {
                    if let Some(s_id) = student_tokenizer.token_to_id(&t1) {
                        if s_id as usize != i {
                            is_identity = false;
                            break;
                        }
                    } else {
                        is_identity = false;
                        break;
                    }
                }
            }
        }

        if is_identity {
            println!("    [+] Detectada identidad de vocabulario. Saltando mapeo exhaustivo.");
            for i in 0..vocab_size {
                vocab_mapping[i] = Some(i);
            }
        } else {
            // Paralelizar con Rayon para evitar bloqueos en dispositivos móviles
            use rayon::prelude::*;
            let results: Vec<Option<usize>> = (0..vocab_size)
                .into_par_iter()
                .map(|i| {
                    if let Ok(token_str) = tokenizer.decode(&[i as u32], true) {
                        if !token_str.is_empty() {
                            return student_tokenizer
                                .token_to_id(&token_str)
                                .map(|id| id as usize);
                        }
                    }
                    None
                })
                .collect();
            vocab_mapping = results;
            println!("    [✔] Mapeo completado (Rayon).");
        }

        Ok(Teacher {
            name,
            model,
            tokenizer,
            vocab_mapping,
            is_identity_vocab: is_identity,
        })
    }

    /// Crea un maestro a partir de un modelo GenomicLLM y su tokenizador ya cargados.
    pub fn from_model(
        name: String,
        model: GenomicLLM,
        tokenizer: GajeTokenizer,
        student_tokenizer: &GajeTokenizer,
    ) -> Self {
        let vocab_size = tokenizer.vocab_size();
        let mut vocab_mapping = vec![None; vocab_size];

        let mut is_identity = vocab_size == student_tokenizer.vocab_size();
        if is_identity {
            for i in 0..100.min(vocab_size) {
                if let Ok(t1) = tokenizer.decode(&[i as u32], true) {
                    if let Some(s_id) = student_tokenizer.token_to_id(&t1) {
                        if s_id as usize != i {
                            is_identity = false;
                            break;
                        }
                    } else {
                        is_identity = false;
                        break;
                    }
                }
            }
        }

        if is_identity {
            for i in 0..vocab_size {
                vocab_mapping[i] = Some(i);
            }
        } else {
            use rayon::prelude::*;
            let results: Vec<Option<usize>> = (0..vocab_size)
                .into_par_iter()
                .map(|i| {
                    if let Ok(token_str) = tokenizer.decode(&[i as u32], true) {
                        if !token_str.is_empty() {
                            return student_tokenizer
                                .token_to_id(&token_str)
                                .map(|id| id as usize);
                        }
                    }
                    None
                })
                .collect();
            vocab_mapping = results;
        }

        Teacher {
            name,
            model,
            tokenizer,
            vocab_mapping,
            is_identity_vocab: is_identity,
        }
    }
}

