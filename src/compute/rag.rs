use rayon::prelude::*;
use std::sync::Arc;

#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(not(feature = "python"))]
use crate::pyo3_shim::exceptions::PyValueError;
#[cfg(not(feature = "python"))]
use crate::pyo3_shim::*;

/// 🧬 NativeSemanticRAG: Motor de Recuperación Aumentada por Generación Semántica Nativa en Rust.
/// Realiza búsquedas de similitud coseno ultrarrápidas y filtrado en memoria compartida.
#[cfg_attr(feature = "python", pyclass)]
#[derive(Debug, Clone)]
pub struct NativeSemanticRAG {
    pub documents: Vec<String>,
    pub embeddings: Vec<Vec<f32>>,
}

#[cfg_attr(feature = "python", pymethods)]
impl NativeSemanticRAG {
    #[cfg(feature = "python")]
    #[new]
    pub fn new() -> Self {
        Self {
            documents: Vec::new(),
            embeddings: Vec::new(),
        }
    }

    pub fn add_document(&mut self, text: String, embedding: Vec<f32>) -> PyResult<()> {
        if embedding.is_empty() {
            return Err(PyValueError::new_err("Embedding vector cannot be empty"));
        }
        self.documents.push(text);
        self.embeddings.push(embedding);
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.documents.len()
    }

    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    pub fn clear(&mut self) {
        self.documents.clear();
        self.embeddings.clear();
    }

    pub fn search(&self, query_embedding: Vec<f32>, top_k: usize) -> PyResult<Vec<(String, f32)>> {
        if query_embedding.is_empty() {
            return Err(PyValueError::new_err("Query embedding vector cannot be empty"));
        }
        if self.documents.is_empty() {
            return Ok(Vec::new());
        }

        let query = Arc::new(query_embedding);
        let q_norm = query.iter().map(|&v| v * v).sum::<f32>().sqrt();
        if q_norm == 0.0 {
            return Ok(Vec::new());
        }

        let mut scores: Vec<(usize, f32)> = self
            .embeddings
            .par_iter()
            .enumerate()
            .map(|(idx, emb)| {
                if emb.len() != query.len() {
                    return (idx, -1.0f32);
                }
                let dot: f32 = emb.iter().zip(query.iter()).map(|(&a, &b)| a * b).sum();
                let e_norm: f32 = emb.iter().map(|&v| v * v).sum::<f32>().sqrt();
                let sim = if e_norm == 0.0 { 0.0 } else { dot / (q_norm * e_norm) };
                (idx, sim)
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let results: Vec<(String, f32)> = scores
            .into_iter()
            .take(top_k)
            .map(|(idx, sim)| (self.documents[idx].clone(), sim))
            .collect();

        Ok(results)
    }

    pub fn format_context(&self, retrieved: Vec<(String, f32)>) -> String {
        retrieved
            .into_iter()
            .map(|(doc, _score)| doc)
            .collect::<Vec<String>>()
            .join("\n---\n")
    }
}
