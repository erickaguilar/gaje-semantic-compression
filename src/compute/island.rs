//! # 🏝️ Island Model: Orquestador de Nichos Semánticos de Memoria
//!
//! Este módulo implementa el orquestador de memoria persistente distribuida en islas:
//! - **Episódica**: Eventos y acciones recientes.
//! - **Documental**: Base de conocimiento de referencia rápida.
//! - **Conversacional**: Historial de diálogo y contexto de sesión activo.

use crate::io::gmem::GmemMemoryIndex;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum IslandNiche {
    Episodic,
    Documental,
    Conversational,
}

pub struct IslandSearchResult {
    pub niche: IslandNiche,
    pub id: u64,
    pub similarity: f32,
    pub text: String,
}

pub struct IslandOrchestrator {
    pub dim: u32,
    pub episodic: GmemMemoryIndex,
    pub documental: GmemMemoryIndex,
    pub conversational: GmemMemoryIndex,
}

impl IslandOrchestrator {
    pub fn new(dim: u32) -> Self {
        Self {
            dim,
            episodic: GmemMemoryIndex::new(dim),
            documental: GmemMemoryIndex::new(dim),
            conversational: GmemMemoryIndex::new(dim),
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
                similarity: sim,
                text: entry.text.clone(),
            });
        }
        for (entry, sim) in res_doc {
            results.push(IslandSearchResult {
                niche: IslandNiche::Documental,
                id: entry.id,
                similarity: sim,
                text: entry.text.clone(),
            });
        }
        for (entry, sim) in res_conv {
            results.push(IslandSearchResult {
                niche: IslandNiche::Conversational,
                id: entry.id,
                similarity: sim,
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
            if m.similarity > 0.65 {
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
