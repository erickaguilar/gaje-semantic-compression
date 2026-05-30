/// 🧬 SessionBuffer: Memoria Intermedia Toroidal para GAJE-Flow
/// Implementa un Ring Buffer basado en SoA (Structure of Arrays) para optimizar
/// la gestión de memoria en hardware ARM y permitir la recirculación semántica.

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::exceptions::{PyIOError, PyValueError};

#[cfg(not(feature = "python"))]
use crate::pyo3_shim::*;
#[cfg(not(feature = "python"))]
use crate::pyo3_shim::exceptions::{PyIOError, PyValueError};

use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};

/// 🧬 SessionBuffer: Memoria Intermedia Toroidal para GAJE-Flow
/// Implementa un Ring Buffer basado en SoA (Structure of Arrays) para optimizar
/// la gestión de memoria en hardware ARM y permitir la recirculación semántica.
#[cfg_attr(feature = "python", pyclass)]
#[derive(Serialize, Deserialize)]
pub struct SessionBuffer {
    capacity: usize,
    dim: usize,
    head: usize,
    is_full: bool,
    /// Almacenamiento contiguo de vectores de fase (SoA)
    phases: Vec<f32>,
    /// Timestamps de cada interacción
    timestamps: Vec<u64>,
    /// Texto comprimido de las interacciones
    texts: Vec<String>,
}

#[cfg_attr(feature = "python", pymethods)]
impl SessionBuffer {
    /// Crea un nuevo buffer de sesión con la capacidad y dimensión especificadas.
    #[cfg(feature = "python")]
    #[new]
    pub fn py_new(capacity: usize, dim: usize) -> Self {
        Self::new(capacity, dim)
    }

    /// Inserta una nueva interacción en el buffer circular.
    pub fn push(&mut self, text: String, phase: Vec<f32>, timestamp: u64) {
        if phase.len() != self.dim {
            return;
        }

        let offset = self.head * self.dim;
        self.phases[offset..offset + self.dim].copy_from_slice(&phase);
        self.timestamps[self.head] = timestamp;
        self.texts[self.head] = text;

        self.head += 1;
        if self.head >= self.capacity {
            self.head = 0;
            self.is_full = true;
        }
    }

    /// Recupera las interacciones más relevantes basándose en la similitud de fase.
    pub fn retrieve_relevant(&self, query: Vec<f32>, top_k: usize) -> PyResult<Vec<String>> {
        if query.len() != self.dim {
            return Ok(Vec::new());
        }

        let limit = if self.is_full { self.capacity } else { self.head };
        if limit == 0 {
            return Ok(Vec::new());
        }

        let mut scores: Vec<(usize, f32)> = (0..limit)
            .map(|i| {
                let offset = i * self.dim;
                let score = self.dot_product(&query, &self.phases[offset..offset + self.dim]);
                (i, score)
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let results = scores
            .into_iter()
            .take(top_k)
            .map(|(i, _)| self.texts[i].clone())
            .collect();
            
        Ok(results)
    }

    /// Guarda el buffer en un archivo binario.
    pub fn dump_to_disk(&self, filepath: String) -> PyResult<()> {
        let file = File::create(filepath).map_err(|e| PyIOError::new_err(e.to_string()))?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, &self).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(())
    }

    /// Carga el buffer desde un archivo binario.
    #[staticmethod]
    pub fn load_from_disk(filepath: String) -> PyResult<Self> {
        let file = File::open(filepath).map_err(|e| PyIOError::new_err(e.to_string()))?;
        let reader = BufReader::new(file);
        let buffer: SessionBuffer = bincode::deserialize_from(reader).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(buffer)
    }

    pub fn __len__(&self) -> usize {
        self.len()
    }
}

impl SessionBuffer {
    /// Crea un nuevo buffer de sesión con la capacidad y dimensión especificadas.
    pub fn new(capacity: usize, dim: usize) -> Self {
        Self {
            capacity,
            dim,
            head: 0,
            is_full: false,
            phases: vec![0.0; capacity * dim],
            timestamps: vec![0; capacity],
            texts: vec![String::new(); capacity],
        }
    }

    /// Cálculo optimizado del producto punto.
    #[inline(always)]
    fn dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Obtiene el número actual de elementos en el buffer.
    pub fn len(&self) -> usize {
        if self.is_full {
            self.capacity
        } else {
            self.head
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
