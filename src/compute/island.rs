//! # 🏝️ Island Model: Orquestador de Nichos Semánticos de Memoria
//!
//! Este módulo implementa el orquestador de memoria persistente distribuida en islas:
//! - **Episódica**: Eventos y acciones recientes.
//! - **Documental**: Base de conocimiento de referencia rápida.
//! - **Conversacional**: Historial de diálogo y contexto de sesión activo.

use crate::core::gtok::ChatTemplate;
use crate::core::tokenizer::GajeTokenizer;
use crate::io::gmem::GmemMemoryIndex;
use crate::nn::llm::GenomicLLM;
use std::path::{Path, PathBuf};
use std::time::Instant;
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

/// Mensaje de diálogo estructurado para templates de chat
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ChatMessage {
    pub role: Option<String>,
    pub content: Option<String>,
    pub message: Option<String>,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Some(role.into()),
            content: Some(content.into()),
            message: None,
        }
    }

    pub fn text(&self) -> &str {
        self.content
            .as_deref()
            .or(self.message.as_deref())
            .unwrap_or("")
    }

    pub fn role_str(&self) -> &str {
        self.role.as_deref().unwrap_or("user")
    }
}

/// Los 6 estados canónicos de decisión del subsistema de memoria hipocampal
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum MemoryDecision {
    #[serde(rename = "memory_disabled")]
    Disabled,
    #[serde(rename = "memory_empty")]
    Empty,
    #[serde(rename = "memory_dim_mismatch")]
    DimMismatch { expected: usize, found: usize },
    #[serde(rename = "memory_injected")]
    Injected {
        facts_count: usize,
        top_sim: f32,
        delta_top: f32,
    },
    #[serde(rename = "rejected_low_similarity")]
    RejectedLowSimilarity {
        top_sim: f32,
        threshold: f32,
    },
    #[serde(rename = "rejected_entropy_gap")]
    RejectedEntropyGap {
        top_sim: f32,
        delta_top: f32,
        threshold: f32,
    },
}

impl MemoryDecision {
    pub fn state_str(&self) -> &'static str {
        match self {
            MemoryDecision::Disabled => "memory_disabled",
            MemoryDecision::Empty => "memory_empty",
            MemoryDecision::DimMismatch { .. } => "memory_dim_mismatch",
            MemoryDecision::Injected { .. } => "memory_injected",
            MemoryDecision::RejectedLowSimilarity { .. } => "rejected_low_similarity",
            MemoryDecision::RejectedEntropyGap { .. } => "rejected_entropy_gap",
        }
    }
}

/// Telemetría completa del subsistema de memoria para SSE, HTTP y REPL
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemoryTelemetry {
    pub decision: MemoryDecision,
    pub state: String,
    pub latency_ms: f32,
    pub whitening_active: bool,
    pub whitening_missing: bool,
    pub facts_injected: usize,
    pub retrieved_facts: Vec<String>,
}

/// Configuración de inferencia con memoria hipocampal
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemoryConfig {
    pub use_memory: bool,
    pub threshold: f32,
    pub uses_whitening: bool,
    pub mu_vector: Option<Vec<f32>>,
    pub whitening_missing: bool,
    pub max_tokens_context: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            use_memory: false,
            threshold: 0.50,
            uses_whitening: false,
            mu_vector: None,
            whitening_missing: false,
            max_tokens_context: 128,
        }
    }
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
        // Si Delta_top < entropy_gap_threshold, se clasifica como búsqueda difusa / colisión ambigua
        // y se aborta la inyección para proteger al transformador de alucinaciones inducidas.
        let top_sim = matches[0].similarity;
        if matches.len() >= 2 {
            let second_sim = matches[1].similarity;
            let delta_top = top_sim - second_sim;
            if delta_top < self.entropy_gap_threshold {
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

    /// Ensambla el bloque de sistema enriquecido con recuerdos episódicos/documentales/conversacionales
    /// aplicando el filtro de umbral global, Entropy Gap y poda competitiva K-WTA.
    pub fn build_augmented_system_prompt(
        &self,
        system_prompt: &str,
        matches: &[IslandSearchResult],
        max_tokens_context: usize,
    ) -> (String, MemoryDecision, Vec<String>) {
        self.build_augmented_system_prompt_with_threshold(
            system_prompt,
            matches,
            self.min_similarity,
            max_tokens_context,
        )
    }

    /// Ensambla el bloque de sistema enriquecido especificando un umbral tau personalizado
    pub fn build_augmented_system_prompt_with_threshold(
        &self,
        system_prompt: &str,
        matches: &[IslandSearchResult],
        threshold: f32,
        max_tokens_context: usize,
    ) -> (String, MemoryDecision, Vec<String>) {
        if matches.is_empty() {
            return (
                system_prompt.to_string(),
                MemoryDecision::RejectedLowSimilarity {
                    top_sim: 0.0,
                    threshold,
                },
                Vec::new(),
            );
        }

        let top_sim = matches[0].similarity;
        if top_sim < threshold {
            return (
                system_prompt.to_string(),
                MemoryDecision::RejectedLowSimilarity {
                    top_sim,
                    threshold,
                },
                Vec::new(),
            );
        }

        let delta_top = if matches.len() >= 2 {
            top_sim - matches[1].similarity
        } else {
            1.0
        };

        if matches.len() >= 2 && delta_top < self.entropy_gap_threshold {
            return (
                system_prompt.to_string(),
                MemoryDecision::RejectedEntropyGap {
                    top_sim,
                    delta_top,
                    threshold: self.entropy_gap_threshold,
                },
                Vec::new(),
            );
        }

        let kwta_cutoff = top_sim * self.kwta_ratio;
        let mut retrieved_facts = Vec::new();
        let mut total_chars = 0;
        let max_chars = max_tokens_context * 4;

        let doc_min = if threshold < 0.65 {
            (threshold + 0.10).min(0.85)
        } else {
            self.documental_min_sim.max(threshold)
        };
        let epi_min = threshold.min(self.episodic_min_sim);

        for m in matches {
            let niche_min = match m.niche {
                IslandNiche::Documental => doc_min,
                IslandNiche::Episodic | IslandNiche::Conversational => epi_min,
            };
            if m.similarity >= niche_min
                && m.similarity >= threshold
                && m.similarity >= kwta_cutoff
            {
                let prefix = match m.niche {
                    IslandNiche::Episodic => "[Memoria Episódica]",
                    IslandNiche::Documental => "[Conocimiento Base]",
                    IslandNiche::Conversational => "[Historial Previo]",
                };
                let snippet = format!("- {} {}", prefix, m.text.trim());
                if total_chars + snippet.len() > max_chars && !retrieved_facts.is_empty() {
                    break;
                }
                total_chars += snippet.len();
                retrieved_facts.push(snippet);
            }
        }

        if retrieved_facts.is_empty() {
            return (
                system_prompt.to_string(),
                MemoryDecision::RejectedLowSimilarity {
                    top_sim,
                    threshold,
                },
                Vec::new(),
            );
        }

        let knowledge_block = format!(
            "[Conocimiento Recuperado:\n{}]",
            retrieved_facts.join("\n")
        );
        let effective_sys = if system_prompt.trim().is_empty() {
            knowledge_block
        } else {
            format!("{}\n\n{}", system_prompt.trim(), knowledge_block)
        };

        (
            effective_sys,
            MemoryDecision::Injected {
                facts_count: retrieved_facts.len(),
                top_sim,
                delta_top,
            },
            retrieved_facts,
        )
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
            let idx = crate::io::gmem::GmemMemoryIndex::load_from_file(&epi_path)?;
            if idx.header.dim != self.dim && idx.header.num_entries > 0 {
                eprintln!(
                    "⚠️ [Island Memory] Dimensión no coincidente en {}: archivo dim={}, modelo dim={}",
                    epi_path, idx.header.dim, self.dim
                );
            }
            self.episodic = idx;
        }
        if std::path::Path::new(&doc_path).exists() {
            let idx = crate::io::gmem::GmemMemoryIndex::load_from_file(&doc_path)?;
            if idx.header.dim != self.dim && idx.header.num_entries > 0 {
                eprintln!(
                    "⚠️ [Island Memory] Dimensión no coincidente en {}: archivo dim={}, modelo dim={}",
                    doc_path, idx.header.dim, self.dim
                );
            }
            self.documental = idx;
        }
        if std::path::Path::new(&conv_path).exists() {
            let idx = crate::io::gmem::GmemMemoryIndex::load_from_file(&conv_path)?;
            if idx.header.dim != self.dim && idx.header.num_entries > 0 {
                eprintln!(
                    "⚠️ [Island Memory] Dimensión no coincidente en {}: archivo dim={}, modelo dim={}",
                    conv_path, idx.header.dim, self.dim
                );
            }
            self.conversational = idx;
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

    /// Resuelve la ruta canónica del directorio de memoria para un modelo
    pub fn resolve_memory_dir(model_path: &Path) -> PathBuf {
        let parent = model_path.parent().unwrap_or_else(|| Path::new("."));
        let stem = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("default");

        // Candidato 1: <parent>/<stem>_memory (e.g. models/born/max_laser_memory)
        let c1 = parent.join(format!("{}_memory", stem));
        if c1.is_dir() {
            return c1;
        }

        // Candidato 2: models/memory/<stem>
        let c2 = PathBuf::from("models/memory").join(stem);
        if c2.is_dir() {
            return c2;
        }

        // Candidato 3: data/memory/<stem>
        let c3 = PathBuf::from("data/memory").join(stem);
        if c3.is_dir() {
            return c3;
        }

        // Por defecto para inicialización o persistencia
        c1
    }

    /// Carga el orquestador de memoria vinculado al modelo o inicializa uno nuevo
    pub fn load_paired_for_model(
        model_path: &Path,
        dim: u32,
        threshold: f32,
    ) -> (Self, PathBuf) {
        let mem_dir = Self::resolve_memory_dir(model_path);
        let mut orch = Self::new(dim);
        orch.min_similarity = threshold;

        // Calibrar submárgenes por nicho según el umbral global del modelo
        if threshold < 0.65 {
            orch.documental_min_sim = (threshold + 0.10).min(0.85);
            orch.episodic_min_sim = threshold;
        } else if threshold > 0.70 {
            orch.documental_min_sim = (threshold + 0.08).min(0.92);
            orch.episodic_min_sim = threshold;
        }

        if mem_dir.is_dir() {
            let dir_str = mem_dir.to_string_lossy();
            if let Err(e) = orch.load_all(&dir_str) {
                eprintln!("⚠️ [Island Memory] Error leyendo memoria en {:?}: {}", mem_dir, e);
            } else {
                let total = orch.episodic.entries.len()
                    + orch.documental.entries.len()
                    + orch.conversational.entries.len();
                if total > 0 {
                    println!(
                        "🧠 [Island Memory] Vinculados {} recuerdos desde {:?} (E:{}, D:{}, C:{})",
                        total,
                        mem_dir,
                        orch.episodic.entries.len(),
                        orch.documental.entries.len(),
                        orch.conversational.entries.len()
                    );
                }
            }
        }

        (orch, mem_dir)
    }

    /// Intenta localizar y cargar el directorio de memoria asociado a un modelo
    pub fn try_load_paired_memory(model_path: &str, dim: u32) -> Option<Self> {
        let p = Path::new(model_path);
        let (orch, mem_dir) = Self::load_paired_for_model(p, dim, 0.65);
        if mem_dir.is_dir() {
            let total = orch.episodic.entries.len()
                + orch.documental.entries.len()
                + orch.conversational.entries.len();
            if total > 0 {
                return Some(orch);
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

/// Detección canónica de la plantilla de diálogo por introspección de tokens de parada o vocabulario
pub fn detect_chat_template_from_tokenizer(tokenizer: &GajeTokenizer) -> ChatTemplate {
    if let Some(gtok) = tokenizer.gtok() {
        gtok.detect_chat_template()
    } else if tokenizer.token_to_id("<|im_start|>").is_some() {
        ChatTemplate::ChatML
    } else if tokenizer.token_to_id("<|start_header_id|>").is_some() {
        ChatTemplate::Llama3
    } else if tokenizer.token_to_id("<start_of_turn>").is_some() {
        ChatTemplate::Gemma
    } else if tokenizer.token_to_id("[INST]").is_some() {
        ChatTemplate::Llama2
    } else if tokenizer.token_to_id("<|system|>").is_some() {
        ChatTemplate::Phi3
    } else {
        ChatTemplate::Classic
    }
}

/// Formatea un prompt de chat aplicando la plantilla de diálogo adecuada al organismo
pub fn format_chat_prompt_from_template(
    template: ChatTemplate,
    effective_sys_prompt: &str,
    user_msg: &str,
    history: Option<&[ChatMessage]>,
) -> String {
    let mut full_prompt = String::new();
    match template {
        ChatTemplate::ChatML => {
            if !effective_sys_prompt.is_empty() {
                full_prompt.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", effective_sys_prompt));
            }
            if let Some(hist) = history {
                for msg in hist.iter().rev().take(6).rev() {
                    let role = msg.role_str();
                    let content = msg.text();
                    full_prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, content));
                }
            }
            full_prompt.push_str(&format!(
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                user_msg.trim()
            ));
        }
        ChatTemplate::Llama3 => {
            full_prompt.push_str("<|begin_of_text|>");
            if !effective_sys_prompt.is_empty() {
                full_prompt.push_str(&format!(
                    "<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>",
                    effective_sys_prompt
                ));
            }
            if let Some(hist) = history {
                for msg in hist.iter().rev().take(6).rev() {
                    let role = msg.role_str();
                    let content = msg.text();
                    full_prompt.push_str(&format!(
                        "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                        role, content
                    ));
                }
            }
            full_prompt.push_str(&format!(
                "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                user_msg.trim()
            ));
        }
        ChatTemplate::Llama2 => {
            full_prompt.push_str("<s>[INST] ");
            if !effective_sys_prompt.is_empty() {
                full_prompt.push_str(&format!("<<SYS>>\n{}\n<</SYS>>\n\n", effective_sys_prompt));
            }
            if let Some(hist) = history {
                for msg in hist.iter().rev().take(6).rev() {
                    let role = msg.role_str();
                    let content = msg.text();
                    if role == "assistant" {
                        full_prompt.push_str(&format!("{} </s><s>[INST] ", content));
                    } else {
                        full_prompt.push_str(&format!("{} [/INST] ", content));
                    }
                }
            }
            full_prompt.push_str(&format!("{} [/INST] ", user_msg.trim()));
        }
        ChatTemplate::Gemma => {
            if let Some(hist) = history {
                for msg in hist.iter().rev().take(6).rev() {
                    let role = if msg.role_str() == "assistant" { "model" } else { "user" };
                    let content = msg.text();
                    full_prompt.push_str(&format!("<start_of_turn>{}\n{}<end_of_turn>\n", role, content));
                }
            }
            if !effective_sys_prompt.is_empty() {
                full_prompt.push_str(&format!(
                    "<start_of_turn>user\n{}\n\n{}<end_of_turn>\n<start_of_turn>model\n",
                    effective_sys_prompt,
                    user_msg.trim()
                ));
            } else {
                full_prompt.push_str(&format!(
                    "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n",
                    user_msg.trim()
                ));
            }
        }
        ChatTemplate::Phi3 => {
            if !effective_sys_prompt.is_empty() {
                full_prompt.push_str(&format!("<|system|>\n{}<|end|>\n", effective_sys_prompt));
            }
            if let Some(hist) = history {
                for msg in hist.iter().rev().take(6).rev() {
                    let role = msg.role_str();
                    let content = msg.text();
                    full_prompt.push_str(&format!("<|{}|>\n{}<|end|>\n", role, content));
                }
            }
            full_prompt.push_str(&format!(
                "<|user|>\n{}<|end|>\n<|assistant|>\n",
                user_msg.trim()
            ));
        }
        _ => {
            if !effective_sys_prompt.is_empty() {
                full_prompt.push_str(&format!("System: {}\n\n", effective_sys_prompt));
            }
            if let Some(hist) = history {
                for msg in hist.iter().rev().take(6).rev() {
                    let role = if msg.role_str() == "assistant" { "Assistant" } else { "User" };
                    let content = msg.text();
                    full_prompt.push_str(&format!("{}: {}\n", role, content));
                }
            }
            full_prompt.push_str(&format!("User: {}\nAssistant: ", user_msg.trim()));
        }
    }
    full_prompt
}

/// Descubre la ruta del archivo satélite .mu.bin asociado al modelo
pub fn resolve_mu_vector_path(model_path: &Path) -> Option<PathBuf> {
    let stem = model_path.file_stem()?.to_str()?;

    // Candidato 1: Mismo directorio que el modelo, con extensión .mu.bin
    let c1 = model_path.with_extension("mu.bin");
    if c1.is_file() {
        return Some(c1);
    }

    // Candidato 2: <parent>/<stem>.mu.bin
    if let Some(parent) = model_path.parent() {
        let c2 = parent.join(format!("{}.mu.bin", stem));
        if c2.is_file() {
            return Some(c2);
        }
    }

    // Candidato 3: models/production/<stem>.mu.bin
    let c3 = PathBuf::from("models/production").join(format!("{}.mu.bin", stem));
    if c3.is_file() {
        return Some(c3);
    }

    // Candidato 4: models/born/<stem>.mu.bin
    let c4 = PathBuf::from("models/born").join(format!("{}.mu.bin", stem));
    if c4.is_file() {
        return Some(c4);
    }

    // Candidato 5: data/calibration/<stem>.mu.bin
    let c5 = PathBuf::from("data/calibration").join(format!("{}.mu.bin", stem));
    if c5.is_file() {
        return Some(c5);
    }

    None
}

/// Carga el vector satélite μ en f32 little-endian verificando dimensión
pub fn load_mu_vector(mu_path: &Path, expected_dim: usize) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(mu_path)
        .map_err(|e| format!("Error leyendo {}: {}", mu_path.display(), e))?;
    if bytes.len() != expected_dim * 4 {
        return Err(format!(
            "Dimensión inválida en .mu.bin: se esperaban {} bytes (dim={}), encontrados {} bytes",
            expected_dim * 4,
            expected_dim,
            bytes.len()
        ));
    }
    let mut mu = Vec::with_capacity(expected_dim);
    for chunk in bytes.chunks_exact(4) {
        let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if val.is_nan() || val.is_infinite() {
            return Err("El archivo .mu.bin contiene valores NaN o Inf".to_string());
        }
        mu.push(val);
    }
    Ok(mu)
}

/// Configura umbrales calibrados y vector satélite μ según el modelo (Opción C de fallback)
pub fn configure_model_memory(
    model_path: &Path,
    dim: usize,
) -> (f32, bool, Option<Vec<f32>>, bool) {
    let stem = model_path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .to_lowercase();

    // 1. max.gaje (Llama Q2_0 256d): Weighted pooling nativo, sin whitening, tau*=0.42
    if stem.contains("max") || dim <= 384 {
        return (0.42, false, None, false);
    }

    // 2. Modelos que requieren whitening: Qwen (896d) o Pico/Smol (576d)
    let needs_whitening = stem.contains("qwen") || stem.contains("pico") || stem.contains("smol") || dim > 384;
    let (whitened_tau, unwhitened_tau) = if stem.contains("qwen") || dim >= 896 {
        (0.33, 0.50)
    } else if stem.contains("pico") || stem.contains("smol") {
        (0.27, 0.65)
    } else {
        (0.30, 0.65)
    };

    if needs_whitening {
        if let Some(mu_path) = resolve_mu_vector_path(model_path) {
            match load_mu_vector(&mu_path, dim) {
                Ok(mu) => {
                    println!("🧬 [Island Memory] Vector satélite μ cargado ({:?}, dim={}) -> Whitening activo (τ*={:.2})", mu_path, dim, whitened_tau);
                    return (whitened_tau, true, Some(mu), false);
                }
                Err(e) => {
                    eprintln!("⚠️ [Island Memory] Error leyendo .mu.bin {:?}: {}. Fallback Opción C activo.", mu_path, e);
                    return (unwhitened_tau, false, None, true);
                }
            }
        } else {
            eprintln!("⚠️ [Island Memory] Archivo .mu.bin no encontrado para {:?}. Fallback Opción C activo (sin whitening, τ*={:.2}).", model_path, unwhitened_tau);
            return (unwhitened_tau, false, None, true);
        }
    }

    (0.65, false, None, false)
}

/// Único punto de verdad para recuperar contexto asociativo, aplicar filtros de gating,
/// ensamblar el prompt enriquecido y generar telemetría de 6 estados.
pub fn prepare_prompt_with_memory(
    user_message: &str,
    history: Option<&[ChatMessage]>,
    system_prompt: &str,
    template: ChatTemplate,
    llm: &GenomicLLM,
    tokenizer: &GajeTokenizer,
    memory: Option<&IslandOrchestrator>,
    config: &MemoryConfig,
) -> (String, MemoryTelemetry) {
    // 1. Si la memoria no está habilitada explícitamente:
    if !config.use_memory {
        let prompt = format_chat_prompt_from_template(template, system_prompt, user_message, history);
        return (
            prompt,
            MemoryTelemetry {
                decision: MemoryDecision::Disabled,
                state: "memory_disabled".to_string(),
                latency_ms: 0.0,
                whitening_active: false,
                whitening_missing: config.whitening_missing,
                facts_injected: 0,
                retrieved_facts: Vec::new(),
            },
        );
    }

    let t0 = Instant::now();

    // 2. Verificar existencia y contenido del orquestador de memoria:
    let orch = match memory {
        Some(o) => o,
        None => {
            let prompt = format_chat_prompt_from_template(template, system_prompt, user_message, history);
            return (
                prompt,
                MemoryTelemetry {
                    decision: MemoryDecision::Empty,
                    state: "memory_empty".to_string(),
                    latency_ms: t0.elapsed().as_secs_f64() as f32 * 1000.0,
                    whitening_active: false,
                    whitening_missing: config.whitening_missing,
                    facts_injected: 0,
                    retrieved_facts: Vec::new(),
                },
            );
        }
    };

    let total_entries = orch.episodic.entries.len()
        + orch.documental.entries.len()
        + orch.conversational.entries.len();
    if total_entries == 0 {
        let prompt = format_chat_prompt_from_template(template, system_prompt, user_message, history);
        return (
            prompt,
            MemoryTelemetry {
                decision: MemoryDecision::Empty,
                state: "memory_empty".to_string(),
                latency_ms: t0.elapsed().as_secs_f64() as f32 * 1000.0,
                whitening_active: false,
                whitening_missing: config.whitening_missing,
                facts_injected: 0,
                retrieved_facts: Vec::new(),
            },
        );
    }

    // 3. Verificar congruencia de dimensión:
    let expected_dim = llm.dim();
    let memory_dim = orch.dim as usize;
    if memory_dim != expected_dim {
        let prompt = format_chat_prompt_from_template(template, system_prompt, user_message, history);
        return (
            prompt,
            MemoryTelemetry {
                decision: MemoryDecision::DimMismatch {
                    expected: expected_dim,
                    found: memory_dim,
                },
                state: "memory_dim_mismatch".to_string(),
                latency_ms: t0.elapsed().as_secs_f64() as f32 * 1000.0,
                whitening_active: false,
                whitening_missing: config.whitening_missing,
                facts_injected: 0,
                retrieved_facts: Vec::new(),
            },
        );
    }

    // 4. Extracción de embedding semántico de la consulta:
    let (query_vec, whitening_active) = match (&config.mu_vector, config.uses_whitening) {
        (Some(mu), true) if mu.len() == expected_dim => {
            let raw = llm
                .embed_text(user_message, tokenizer)
                .unwrap_or_else(|_| vec![0.0; expected_dim]);
            let mut centered = vec![0.0f32; expected_dim];
            let mut norm_sq = 0.0f32;
            for i in 0..expected_dim {
                let diff = raw[i] - mu[i];
                centered[i] = diff;
                norm_sq += diff * diff;
            }
            let norm = norm_sq.sqrt().max(1e-8);
            for val in centered.iter_mut() {
                *val /= norm;
            }
            (centered, true)
        }
        _ => {
            let raw = llm
                .embed_text(user_message, tokenizer)
                .unwrap_or_else(|_| vec![0.0; expected_dim]);
            (raw, false)
        }
    };

    // 5. Búsqueda top-k asociativa en islas:
    let matches = orch.retrieve_context(&query_vec, 2);

    // 6. Gating & ensamblado del bloque de sistema enriquecido con umbral de config
    let (effective_sys_prompt, decision, retrieved_facts) = orch
        .build_augmented_system_prompt_with_threshold(
            system_prompt,
            &matches,
            config.threshold,
            config.max_tokens_context,
        );

    // 7. Formateo de plantilla de chat nativa (ChatML, Llama3, etc.)
    let full_prompt = format_chat_prompt_from_template(
        template,
        &effective_sys_prompt,
        user_message,
        history,
    );

    let latency_ms = t0.elapsed().as_secs_f64() as f32 * 1000.0;
    let state_str = decision.state_str().to_string();
    let facts_injected = if let MemoryDecision::Injected { facts_count, .. } = &decision {
        *facts_count
    } else {
        0
    };

    (
        full_prompt,
        MemoryTelemetry {
            decision,
            state: state_str,
            latency_ms,
            whitening_active,
            whitening_missing: config.whitening_missing,
            facts_injected,
            retrieved_facts,
        },
    )
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
        // Delta_top = 0.74 - 0.72 = 0.02 < 0.12 -> Ruido Difuso abortado por Entropy Gap
        let prompt = "¿Pregunta sobre tema desconocido?";
        let augmented = orch.build_augmented_prompt_from_matches(prompt, &matches, 100);
        assert_eq!(augmented, prompt, "Debe abortar la inyección ante ruido difuso");
    }

    #[test]
    fn test_entropy_gap_close_competition_aborted() {
        let orch = IslandOrchestrator::new(4);
        let matches = vec![
            IslandSearchResult {
                niche: IslandNiche::Documental,
                id: 1,
                similarity: 0.92,
                text: "Hecho ambiguo 1".to_string(),
            },
            IslandSearchResult {
                niche: IslandNiche::Documental,
                id: 2,
                similarity: 0.89,
                text: "Hecho ambiguo 2".to_string(),
            },
        ];
        // Delta_top = 0.03 < 0.12 -> Competencia cerrada sin ganador claro: debe abortar
        let prompt = "¿Pregunta clave?";
        let augmented = orch.build_augmented_prompt_from_matches(prompt, &matches, 100);
        assert_eq!(augmented, prompt, "Debe abortar la inyección ante competencia cerrada");

        // Caso con margen claro: Delta_top = 0.92 - 0.70 = 0.22 >= 0.12 -> Ganador claro permitido
        let clear_matches = vec![
            IslandSearchResult {
                niche: IslandNiche::Documental,
                id: 1,
                similarity: 0.92,
                text: "Hecho ganador claro".to_string(),
            },
            IslandSearchResult {
                niche: IslandNiche::Documental,
                id: 2,
                similarity: 0.70,
                text: "Hecho secundario".to_string(),
            },
        ];
        let augmented_clear = orch.build_augmented_prompt_from_matches(prompt, &clear_matches, 100);
        assert!(augmented_clear.contains("Contexto de Memoria Recolectado:"));
        assert!(augmented_clear.contains("Hecho ganador claro"));
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
