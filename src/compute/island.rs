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
    /// Brecha de entropía mínima Delta_top = Sim_1 - Sim_2 para evitar ruido difuso (0.12)
    pub entropy_gap_threshold: f32,
    /// Relación de poda competitiva K-WTA (Sim >= kwta_ratio * Max_Sim, por defecto 0.90)
    pub kwta_ratio: f32,
    /// Umbral estricto para hechos del nicho documental (por defecto 0.82)
    pub documental_min_sim: f32,
    /// Umbral para nicho episódico y conversacional (por defecto 0.70)
    pub episodic_min_sim: f32,
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
            entropy_gap_threshold: 0.12,
            kwta_ratio: 0.90,
            documental_min_sim: 0.82,
            episodic_min_sim: 0.70,
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

    /// Aplica una rotación / transformación ortogonal que desacopla los nichos en R^D
    /// preservando la norma euclidiana del vector (R^T R = I).
    pub fn project_niche_vector(niche: IslandNiche, vector: &[f32]) -> Vec<f32> {
        match niche {
            IslandNiche::Documental => vector.to_vec(),
            IslandNiche::Episodic => {
                // R_epi: alternancia de signos en índices impares
                vector
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| if i % 2 == 1 { -v } else { v })
                    .collect()
            }
            IslandNiche::Conversational => {
                // R_conv: alternancia de signos en bloques de 2 (patrón Walsh-Hadamard)
                vector
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| if (i / 2) % 2 == 1 { -v } else { v })
                    .collect()
            }
        }
    }

    /// Registra un nuevo recuerdo aplicando desacoplamiento ortogonal según el nicho
    pub fn add_memory_orthogonal(&mut self, niche: IslandNiche, id: u64, vector: &[f32], text: String) {
        let projected = Self::project_niche_vector(niche, vector);
        self.add_memory(niche, id, projected, text);
    }

    /// Recupera contexto relevante proyectando la consulta ortogonalmente por cada nicho
    pub fn retrieve_context_orthogonal(
        &self,
        query_vector: &[f32],
        k_per_niche: usize,
    ) -> Vec<IslandSearchResult> {
        let v_doc = Self::project_niche_vector(IslandNiche::Documental, query_vector);
        let v_epi = Self::project_niche_vector(IslandNiche::Episodic, query_vector);
        let v_conv = Self::project_niche_vector(IslandNiche::Conversational, query_vector);

        let (res_epi, (res_doc, res_conv)) = rayon::join(
            || self.episodic.search_top_k(&v_epi, k_per_niche),
            || {
                rayon::join(
                    || self.documental.search_top_k(&v_doc, k_per_niche),
                    || self.conversational.search_top_k(&v_conv, k_per_niche),
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

        // 1. Entropy Gap Gating:
        // Si hay al menos 2 matches, evaluar Delta_top = Sim_1 - Sim_2.
        // Si Delta_top < entropy_gap_threshold && Sim_1 < 0.85, se clasifica como búsqueda difusa / ruido
        // y se aborta la inyección para proteger al transformador de alucinaciones inducidas.
        let top_sim = matches[0].similarity;
        if matches.len() >= 2 {
            let second_sim = matches[1].similarity;
            let delta_top = top_sim - second_sim;
            if delta_top < self.entropy_gap_threshold && top_sim < 0.85 {
                return prompt.to_string();
            }
        }

        // 2. Inhibición Lateral K-WTA: podar cualquier recuerdo que no alcance kwta_ratio * Max_Sim
        let kwta_cutoff = top_sim * self.kwta_ratio;

        let mut context_snippets = Vec::new();
        let mut char_count = 0;
        let max_chars = max_tokens_context * 4; // Aproximación estándar 1 token ~ 4 chars

        for m in matches {
            // Umbral estricto por nicho
            let niche_min = match m.niche {
                IslandNiche::Documental => self.documental_min_sim,
                IslandNiche::Episodic | IslandNiche::Conversational => self.episodic_min_sim,
            };

            if m.similarity >= niche_min
                && m.similarity >= self.min_similarity
                && m.similarity >= kwta_cutoff
            {
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

    /// Genera la cadena de contexto enriquecida utilizando desacoplamiento ortogonal
    pub fn build_augmented_prompt_orthogonal(
        &self,
        prompt: &str,
        query_vector: &[f32],
        max_tokens_context: usize,
    ) -> String {
        let matches = self.retrieve_context_orthogonal(query_vector, 2);
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

        let eval_loss = |weights: &[f32; 3], _min_sim: f32| -> f32 {
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

    /// Genera vector semántico determinista en R^D normalizado
    pub fn vector_from_text(text: &str, dim: usize) -> Vec<f32> {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        text.hash(&mut hasher);
        let seed = hasher.finish();

        let mut vec = vec![0.0f32; dim];
        let mut norm_sq = 0.0f32;
        for (i, val) in vec.iter_mut().enumerate() {
            let pseudo = ((seed
                .wrapping_add(i as u64)
                .wrapping_mul(6364136223846793005))
                >> 32) as i32;
            let f = (pseudo as f32) / (i32::MAX as f32);
            *val = f;
            norm_sq += f * f;
        }
        let norm = norm_sq.sqrt().max(1e-8);
        for val in vec.iter_mut() {
            *val /= norm;
        }
        vec
    }

    /// Intenta localizar y cargar el directorio de memoria asociado a un modelo
    pub fn try_load_paired_memory(model_path: &str, dim: u32) -> Option<Self> {
        let memory_dir = if model_path.ends_with(".gaje") {
            model_path.strip_suffix(".gaje").unwrap().to_string() + "_memory"
        } else if model_path.ends_with(".flat") {
            model_path.strip_suffix(".flat").unwrap().to_string() + "_memory"
        } else {
            format!("{}_memory", model_path)
        };

        if std::path::Path::new(&memory_dir).exists() {
            let mut orch = Self::new(dim);
            if orch.load_all(&memory_dir).is_ok() {
                let total = orch.episodic.entries.len()
                    + orch.documental.entries.len()
                    + orch.conversational.entries.len();
                if total > 0 {
                    return Some(orch);
                }
            }
        }
        None
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
    #[pyo3(signature = (dim, niche_weights=None, min_similarity=None, entropy_gap_threshold=None, kwta_ratio=None, documental_min_sim=None, episodic_min_sim=None))]
    pub fn py_new(
        dim: u32,
        niche_weights: Option<Vec<f32>>,
        min_similarity: Option<f32>,
        entropy_gap_threshold: Option<f32>,
        kwta_ratio: Option<f32>,
        documental_min_sim: Option<f32>,
        episodic_min_sim: Option<f32>,
    ) -> Self {
        let mut orch = Self::new(dim);
        if let Some(w) = niche_weights {
            if w.len() == 3 {
                orch.niche_weights = [w[0], w[1], w[2]];
            }
        }
        if let Some(ms) = min_similarity {
            orch.min_similarity = ms;
        }
        if let Some(eg) = entropy_gap_threshold {
            orch.entropy_gap_threshold = eg;
        }
        if let Some(kr) = kwta_ratio {
            orch.kwta_ratio = kr;
        }
        if let Some(ds) = documental_min_sim {
            orch.documental_min_sim = ds;
        }
        if let Some(es) = episodic_min_sim {
            orch.episodic_min_sim = es;
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

    pub fn add_memory_orthogonal_py(
        &mut self,
        niche: &str,
        id: u64,
        vector: Vec<f32>,
        text: String,
    ) -> PyResult<()> {
        let n = IslandNiche::from_str(niche).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("Nicho inválido: {}", niche))
        })?;
        self.add_memory_orthogonal(n, id, &vector, text);
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

    pub fn retrieve_context_orthogonal_py(
        &self,
        query_vector: Vec<f32>,
        k_per_niche: usize,
    ) -> Vec<(String, u64, f32, String)> {
        self.retrieve_context_orthogonal(&query_vector, k_per_niche)
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

    pub fn build_augmented_prompt_orthogonal_py(
        &self,
        prompt: &str,
        query_vector: Vec<f32>,
        max_tokens_context: usize,
    ) -> String {
        self.build_augmented_prompt_orthogonal(prompt, &query_vector, max_tokens_context)
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

    #[getter]
    pub fn get_entropy_gap_threshold(&self) -> f32 {
        self.entropy_gap_threshold
    }

    #[setter]
    pub fn set_entropy_gap_threshold(&mut self, val: f32) {
        self.entropy_gap_threshold = val;
    }

    #[getter]
    pub fn get_kwta_ratio(&self) -> f32 {
        self.kwta_ratio
    }

    #[setter]
    pub fn set_kwta_ratio(&mut self, val: f32) {
        self.kwta_ratio = val;
    }

    #[getter]
    pub fn get_documental_min_sim(&self) -> f32 {
        self.documental_min_sim
    }

    #[setter]
    pub fn set_documental_min_sim(&mut self, val: f32) {
        self.documental_min_sim = val;
    }

    #[getter]
    pub fn get_episodic_min_sim(&self) -> f32 {
        self.episodic_min_sim
    }

    #[setter]
    pub fn set_episodic_min_sim(&mut self, val: f32) {
        self.episodic_min_sim = val;
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

    #[test]
    fn test_entropy_gap_gating_diffuse_noise() {
        let orch = IslandOrchestrator::new(4);
        let matches = vec![
            IslandSearchResult {
                niche: IslandNiche::Conversational,
                id: 1,
                similarity: 0.74,
                text: "Recuerdo difuso 1".to_string(),
            },
            IslandSearchResult {
                niche: IslandNiche::Conversational,
                id: 2,
                similarity: 0.72,
                text: "Recuerdo difuso 2".to_string(),
            },
        ];
        // Delta_top = 0.74 - 0.72 = 0.02 < 0.12 y top_sim = 0.74 < 0.85 -> Ruido Difuso abortado
        let prompt = "¿Pregunta sobre tema desconocido?";
        let augmented = orch.build_augmented_prompt_from_matches(prompt, &matches, 100);
        assert_eq!(augmented, prompt, "Debe abortar la inyección ante ruido difuso");
    }

    #[test]
    fn test_entropy_gap_high_resonance_override() {
        let orch = IslandOrchestrator::new(4);
        let matches = vec![
            IslandSearchResult {
                niche: IslandNiche::Documental,
                id: 1,
                similarity: 0.92,
                text: "Hecho de alta certeza 1".to_string(),
            },
            IslandSearchResult {
                niche: IslandNiche::Documental,
                id: 2,
                similarity: 0.89,
                text: "Hecho de alta certeza 2".to_string(),
            },
        ];
        // Delta_top = 0.03 < 0.12, pero top_sim = 0.92 >= 0.85 -> Resonancia clara permitida
        let prompt = "¿Pregunta clave?";
        let augmented = orch.build_augmented_prompt_from_matches(prompt, &matches, 100);
        assert!(augmented.contains("Contexto de Memoria Recolectado:"));
        assert!(augmented.contains("Hecho de alta certeza 1"));
    }

    #[test]
    fn test_kwta_competitive_pruning() {
        let orch = IslandOrchestrator::new(4);
        let matches = vec![
            IslandSearchResult {
                niche: IslandNiche::Documental,
                id: 1,
                similarity: 0.95,
                text: "Recuerdo dominante".to_string(),
            },
            IslandSearchResult {
                niche: IslandNiche::Documental,
                id: 2,
                similarity: 0.83, // 0.83 < 0.90 * 0.95 (0.855) -> debe ser podado por K-WTA
                text: "Recuerdo dominado".to_string(),
            },
        ];
        let prompt = "¿Consulta?";
        let augmented = orch.build_augmented_prompt_from_matches(prompt, &matches, 100);
        assert!(augmented.contains("Recuerdo dominante"));
        assert!(!augmented.contains("Recuerdo dominado"), "K-WTA debe podar el recuerdo dominado");
    }

    #[test]
    fn test_orthogonal_subspace_isometry() {
        let v = vec![0.5, -0.5, 0.5, -0.5];
        let norm_orig: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();

        let v_doc = IslandOrchestrator::project_niche_vector(IslandNiche::Documental, &v);
        let v_epi = IslandOrchestrator::project_niche_vector(IslandNiche::Episodic, &v);
        let v_conv = IslandOrchestrator::project_niche_vector(IslandNiche::Conversational, &v);

        let norm_doc: f32 = v_doc.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_epi: f32 = v_epi.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_conv: f32 = v_conv.iter().map(|x| x * x).sum::<f32>().sqrt();

        assert!((norm_doc - norm_orig).abs() < 1e-6);
        assert!((norm_epi - norm_orig).abs() < 1e-6);
        assert!((norm_conv - norm_orig).abs() < 1e-6);
    }
}
