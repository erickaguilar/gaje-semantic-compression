//! # 🏝️ Island Model: Orquestador de Nichos Semánticos de Memoria
//!
//! Este módulo implementa el orquestador de memoria persistente distribuida en islas:
//! - **Episódica**: Eventos y acciones recientes.
//! - **Documental**: Base de conocimiento de referencia rápida.
//! - **Conversacional**: Historial de diálogo y contexto de sesión activo.

use crate::io::gmem::GmemMemoryIndex;
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[derive(Copy, Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum IslandNiche {
    Episodic,
    Documental,
    Conversational,
}

impl IslandNiche {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "episodic" | "episodica" => Some(Self::Episodic),
            "documental" | "doc" => Some(Self::Documental),
            "conversational" | "chat" => Some(Self::Conversational),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Episodic => "episodic",
            Self::Documental => "documental",
            Self::Conversational => "conversational",
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct IslandSearchResult {
    pub niche: IslandNiche,
    pub id: u64,
    pub similarity: f32,
    pub text: String,
}

#[derive(Clone)]
#[cfg_attr(feature = "python", pyclass)]
pub struct IslandOrchestrator {
    pub dim: u32,
    pub episodic: GmemMemoryIndex,
    pub documental: GmemMemoryIndex,
    pub conversational: GmemMemoryIndex,
    pub niche_weights: [f32; 3], // [episodic, documental, conversational]
    pub min_similarity: f32,
}

impl IslandOrchestrator {
    pub fn new(dim: u32) -> Self {
        Self {
            dim,
            episodic: GmemMemoryIndex::new(dim),
            documental: GmemMemoryIndex::new(dim),
            conversational: GmemMemoryIndex::new(dim),
            niche_weights: [1.0, 1.2, 0.8],
            min_similarity: 0.65,
        }
    }

    /// Registra un nuevo recuerdo en la isla seleccionada
    pub fn add_memory(&mut self, niche: IslandNiche, id: u64, vector: Vec<f32>, text: String) {
        match niche {
            IslandNiche::Episodic => self.episodic.add_entry(id, vector, text),
            IslandNiche::Documental => self.documental.add_entry(id, vector, text),
            IslandNiche::Conversational => self.conversational.add_entry(id, vector, text),
        }
    }

    /// Recupera contexto relevante consultando las 3 islas en paralelo mediante Rayon
    pub fn retrieve_context(
        &self,
        query_vector: &[f32],
        k_per_niche: usize,
    ) -> Vec<IslandSearchResult> {
        let (res_epi, (res_doc, res_conv)) = rayon::join(
            || self.episodic.search_top_k(query_vector, k_per_niche),
            || {
                rayon::join(
                    || self.documental.search_top_k(query_vector, k_per_niche),
                    || self.conversational.search_top_k(query_vector, k_per_niche),
                )
            },
        );

        let mut results = Vec::with_capacity(res_epi.len() + res_doc.len() + res_conv.len());

        for (entry, sim) in res_epi {
            results.push(IslandSearchResult {
                niche: IslandNiche::Episodic,
                id: entry.id,
                similarity: sim * self.niche_weights[0],
                text: entry.text.clone(),
            });
        }
        for (entry, sim) in res_doc {
            results.push(IslandSearchResult {
                niche: IslandNiche::Documental,
                id: entry.id,
                similarity: sim * self.niche_weights[1],
                text: entry.text.clone(),
            });
        }
        for (entry, sim) in res_conv {
            results.push(IslandSearchResult {
                niche: IslandNiche::Conversational,
                id: entry.id,
                similarity: sim * self.niche_weights[2],
                text: entry.text.clone(),
            });
        }

        // Ordenar globalmente por similitud decreciente
        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    /// Ensambla el prompt aumentado usando resultados previamente recuperados
    pub fn build_augmented_prompt_from_matches(
        &self,
        prompt: &str,
        matches: &[IslandSearchResult],
        max_tokens_context: usize,
    ) -> String {
        if matches.is_empty() {
            return prompt.to_string();
        }

        let mut context_snippets = Vec::new();
        let mut char_count = 0;
        let max_chars = max_tokens_context * 4; // Aproximación estándar 1 token ~ 4 chars

        for m in matches {
            if m.similarity > self.min_similarity {
                let prefix = match m.niche {
                    IslandNiche::Episodic => "[Memoria Episódica]",
                    IslandNiche::Documental => "[Conocimiento Base]",
                    IslandNiche::Conversational => "[Historial Previo]",
                };
                let snippet = format!("{} {}", prefix, m.text);
                if char_count + snippet.len() > max_chars && !context_snippets.is_empty() {
                    break;
                }
                char_count += snippet.len();
                context_snippets.push(snippet);
            }
        }

        if context_snippets.is_empty() {
            prompt.to_string()
        } else {
            format!(
                "Contexto de Memoria Recolectado:\n{}\n\nPregunta Usuario: {}",
                context_snippets.join("\n"),
                prompt
            )
        }
    }

    /// Genera la cadena de contexto enriquecida para el prefill del LLM
    pub fn build_augmented_prompt(
        &self,
        prompt: &str,
        query_vector: &[f32],
        max_tokens_context: usize,
    ) -> String {
        let matches = self.retrieve_context(query_vector, 2);
        self.build_augmented_prompt_from_matches(prompt, &matches, max_tokens_context)
    }

    /// Optimización SPSA de orden cero para calibrar pesos de nichos de memoria
    pub fn optimize_niche_weights_spsa(
        &mut self,
        queries: &[Vec<f32>],
        target_niche_ids: &[usize], // 0: Episodic, 1: Documental, 2: Conversational
        epochs: usize,
        c: f32,
        lr: f32,
    ) -> f32 {
        if queries.is_empty() || queries.len() != target_niche_ids.len() {
            return 0.0;
        }

        let eval_loss = |weights: &[f32; 3], min_sim: f32| -> f32 {
            let mut loss = 0.0f32;
            for (q, &target_niche) in queries.iter().zip(target_niche_ids.iter()) {
                let (res_epi, (res_doc, res_conv)) = rayon::join(
                    || self.episodic.search_top_k(q, 1),
                    || {
                        rayon::join(
                            || self.documental.search_top_k(q, 1),
                            || self.conversational.search_top_k(q, 1),
                        )
                    },
                );
                let s_epi = res_epi.first().map(|(_, s)| *s * weights[0]).unwrap_or(0.0);
                let s_doc = res_doc.first().map(|(_, s)| *s * weights[1]).unwrap_or(0.0);
                let s_conv = res_conv
                    .first()
                    .map(|(_, s)| *s * weights[2])
                    .unwrap_or(0.0);

                let target_score = match target_niche {
                    0 => s_epi,
                    1 => s_doc,
                    _ => s_conv,
                };
                let max_other = match target_niche {
                    0 => s_doc.max(s_conv),
                    1 => s_epi.max(s_conv),
                    _ => s_epi.max(s_doc),
                };

                // Margin loss: target_score debe superar a otros por margen
                let diff = (max_other - target_score + 0.5).max(0.0);
                loss += diff * diff;
            }
            loss / queries.len() as f32
        };

        let mut current_loss = eval_loss(&self.niche_weights, self.min_similarity);

        for ep in 0..epochs {
            // Generar vector de perturbación Rademacher ±1 independiente por dimensión
            let delta = [
                if ((ep * 1664525 + 1013904223) >> 16) & 1 == 0 {
                    1.0f32
                } else {
                    -1.0f32
                },
                if (((ep + 7) * 22695477 + 1) >> 16) & 1 == 0 {
                    1.0f32
                } else {
                    -1.0f32
                },
                if (((ep + 19) * 1103515245 + 12345) >> 16) & 1 == 0 {
                    1.0f32
                } else {
                    -1.0f32
                },
            ];

            let mut w_plus = self.niche_weights;
            let mut w_minus = self.niche_weights;
            for i in 0..3 {
                w_plus[i] = (w_plus[i] + c * delta[i]).max(0.01);
                w_minus[i] = (w_minus[i] - c * delta[i]).max(0.01);
            }

            let l_plus = eval_loss(&w_plus, self.min_similarity);
            let l_minus = eval_loss(&w_minus, self.min_similarity);

            // Gradiente SPSA
            let g_base = (l_plus - l_minus) / (2.0 * c);
            for i in 0..3 {
                let grad_i = g_base / delta[i];
                self.niche_weights[i] = (self.niche_weights[i] - lr * grad_i).clamp(0.05, 5.0);
            }

            current_loss = eval_loss(&self.niche_weights, self.min_similarity);
        }

        current_loss
    }

    pub fn save_all(&mut self, dir_path: &str) -> std::io::Result<()> {
        std::fs::create_dir_all(dir_path)?;
        // Construir/actualizar el indice IVF de cada isla antes de serializar:
        // el byte index_type=1 y la seccion IVF1 viajan en el archivo resultante.
        // refresh_ivf es no-op bajo el umbral o si ya esta vigente.
        self.episodic.refresh_ivf();
        self.documental.refresh_ivf();
        self.conversational.refresh_ivf();
        self.episodic
            .save_to_file(&format!("{}/episodic.gmem", dir_path))?;
        self.documental
            .save_to_file(&format!("{}/documental.gmem", dir_path))?;
        self.conversational
            .save_to_file(&format!("{}/conversational.gmem", dir_path))?;
        Ok(())
    }

    pub fn save_epoch(
        &mut self,
        dir_path: &str,
        epoch_id: u64,
        parent_epoch: u64,
    ) -> std::io::Result<()> {
        std::fs::create_dir_all(dir_path)?;

        self.episodic.set_epoch_id(epoch_id);
        self.episodic.set_parent_epoch(parent_epoch);
        self.documental.set_epoch_id(epoch_id);
        self.documental.set_parent_epoch(parent_epoch);
        self.conversational.set_epoch_id(epoch_id);
        self.conversational.set_parent_epoch(parent_epoch);

        self.save_all(dir_path)
    }

    pub fn get_epoch_info(&self) -> (u64, u64, bool, bool) {
        (
            self.documental.epoch_id(),
            self.documental.parent_epoch(),
            self.documental.is_sealed(),
            self.documental.is_promoted(),
        )
    }

    pub fn load_all(&mut self, dir_path: &str) -> std::io::Result<()> {
        let epi_path = format!("{}/episodic.gmem", dir_path);
        let doc_path = format!("{}/documental.gmem", dir_path);
        let conv_path = format!("{}/conversational.gmem", dir_path);

        if std::path::Path::new(&epi_path).exists() {
            self.episodic = crate::io::gmem::GmemMemoryIndex::load_from_file(&epi_path)?;
        }
        if std::path::Path::new(&doc_path).exists() {
            self.documental = crate::io::gmem::GmemMemoryIndex::load_from_file(&doc_path)?;
        }
        if std::path::Path::new(&conv_path).exists() {
            self.conversational = crate::io::gmem::GmemMemoryIndex::load_from_file(&conv_path)?;
        }
        Ok(())
    }

    pub fn consolidate_memory(&mut self, dedup_threshold: f32) -> ConsolidationStats {
        let mut transferred_epi = 0;
        let mut transferred_conv = 0;
        let mut pruned = 0;

        let epi_entries = std::mem::take(&mut self.episodic.entries);
        for entry in epi_entries {
            let max_sim = match self.documental.search_top_k(&entry.vector, 1).first() {
                Some((_, s)) => *s,
                None => 0.0,
            };

            if max_sim >= dedup_threshold {
                pruned += 1;
            } else {
                self.documental
                    .add_entry(entry.id, entry.vector, entry.text);
                transferred_epi += 1;
            }
        }

        let conv_entries = std::mem::take(&mut self.conversational.entries);
        for entry in conv_entries {
            let max_sim = match self.documental.search_top_k(&entry.vector, 1).first() {
                Some((_, s)) => *s,
                None => 0.0,
            };

            if max_sim >= dedup_threshold {
                pruned += 1;
            } else {
                self.documental
                    .add_entry(entry.id, entry.vector, entry.text);
                transferred_conv += 1;
            }
        }

        self.documental.set_consolidated(true);
        self.episodic.clear();
        self.conversational.clear();

        ConsolidationStats {
            episodic_transferred: transferred_epi,
            conversational_transferred: transferred_conv,
            duplicates_pruned: pruned,
            total_documental_entries: self.documental.entries.len(),
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ConsolidationStats {
    pub episodic_transferred: usize,
    pub conversational_transferred: usize,
    pub duplicates_pruned: usize,
    pub total_documental_entries: usize,
}

#[cfg(feature = "python")]
#[pymethods]
impl IslandOrchestrator {
    #[new]
    #[pyo3(signature = (dim, niche_weights=None, min_similarity=None))]
    pub fn py_new(dim: u32, niche_weights: Option<Vec<f32>>, min_similarity: Option<f32>) -> Self {
        let mut orch = Self::new(dim);
        if let Some(w) = niche_weights {
            if w.len() == 3 {
                orch.niche_weights = [w[0], w[1], w[2]];
            }
        }
        if let Some(ms) = min_similarity {
            orch.min_similarity = ms;
        }
        orch
    }

    pub fn add_memory_py(
        &mut self,
        niche: &str,
        id: u64,
        vector: Vec<f32>,
        text: String,
    ) -> PyResult<()> {
        let n = IslandNiche::from_str(niche).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("Nicho inválido: {}", niche))
        })?;
        self.add_memory(n, id, vector, text);
        Ok(())
    }

    pub fn retrieve_context_py(
        &self,
        query_vector: Vec<f32>,
        k_per_niche: usize,
    ) -> Vec<(String, u64, f32, String)> {
        self.retrieve_context(&query_vector, k_per_niche)
            .into_iter()
            .map(|r| (r.niche.as_str().to_string(), r.id, r.similarity, r.text))
            .collect()
    }

    pub fn build_augmented_prompt_py(
        &self,
        prompt: &str,
        query_vector: Vec<f32>,
        max_tokens_context: usize,
    ) -> String {
        self.build_augmented_prompt(prompt, &query_vector, max_tokens_context)
    }

    pub fn optimize_spsa_py(
        &mut self,
        queries: Vec<Vec<f32>>,
        target_niche_ids: Vec<usize>,
        epochs: usize,
        c: f32,
        lr: f32,
    ) -> f32 {
        self.optimize_niche_weights_spsa(&queries, &target_niche_ids, epochs, c, lr)
    }

    #[getter]
    pub fn get_niche_weights(&self) -> Vec<f32> {
        self.niche_weights.to_vec()
    }

    #[setter]
    pub fn set_niche_weights(&mut self, weights: Vec<f32>) -> PyResult<()> {
        if weights.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "niche_weights debe tener exactamente 3 elementos [episodic, documental, conversational]"
            ));
        }
        self.niche_weights = [weights[0], weights[1], weights[2]];
        Ok(())
    }

    /// Reconstruye el indice IVF-lite de cada isla (entradas >= umbral).
    /// Devuelve el numero de entradas indexadas por isla.
    pub fn refresh_indexes_py(&mut self) -> PyResult<(usize, usize, usize)> {
        self.episodic.refresh_ivf();
        self.documental.refresh_ivf();
        self.conversational.refresh_ivf();
        Ok((
            self.episodic.entries.len(),
            self.documental.entries.len(),
            self.conversational.entries.len(),
        ))
    }

    pub fn save_all_py(&mut self, dir_path: &str) -> PyResult<()> {
        self.save_all(dir_path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    pub fn save_epoch_py(
        &mut self,
        dir_path: &str,
        epoch_id: u64,
        parent_epoch: u64,
    ) -> PyResult<()> {
        self.save_epoch(dir_path, epoch_id, parent_epoch)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    pub fn get_epoch_info_py(&self) -> (u64, u64, bool, bool) {
        self.get_epoch_info()
    }

    pub fn load_all_py(&mut self, dir_path: &str) -> PyResult<()> {
        self.load_all(dir_path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    pub fn consolidate_memory_py(&mut self, dedup_threshold: f32) -> PyResult<String> {
        let stats = self.consolidate_memory(dedup_threshold);
        serde_json::to_string_pretty(&stats)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_island_orchestrator_retrieval() {
        let mut orch = IslandOrchestrator::new(4);
        orch.add_memory(
            IslandNiche::Documental,
            101,
            vec![1.0, 0.0, 0.0, 0.0],
            "El formato .gmem mapea memoria a 0ms.".to_string(),
        );

        orch.add_memory(
            IslandNiche::Conversational,
            202,
            vec![0.0, 1.0, 0.0, 0.0],
            "El usuario preguntó sobre la capital de Francia.".to_string(),
        );

        let query = vec![0.95, 0.05, 0.0, 0.0];
        let context = orch.build_augmented_prompt("¿Cómo funciona .gmem?", &query, 100);

        assert!(context.contains("Contexto de Memoria Recolectado:"));
        assert!(context.contains("[Conocimiento Base] El formato .gmem mapea memoria a 0ms."));
    }
}
