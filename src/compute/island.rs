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

    /// Recupera contexto relevante consultando las 3 islas en paralelo
    pub fn retrieve_context(
        &self,
        query_vector: &[f32],
        k_per_niche: usize,
    ) -> Vec<IslandSearchResult> {
        let mut results = Vec::new();

        // 1. Consultar Isla Episódica
        for (entry, sim) in self.episodic.search_top_k(query_vector, k_per_niche) {
            results.push(IslandSearchResult {
                niche: IslandNiche::Episodic,
                id: entry.id,
                similarity: sim,
                text: entry.text.clone(),
            });
        }

        // 2. Consultar Isla Documental
        for (entry, sim) in self.documental.search_top_k(query_vector, k_per_niche) {
            results.push(IslandSearchResult {
                niche: IslandNiche::Documental,
                id: entry.id,
                similarity: sim,
                text: entry.text.clone(),
            });
        }

        // 3. Consultar Isla Conversacional
        for (entry, sim) in self.conversational.search_top_k(query_vector, k_per_niche) {
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

    /// Genera la cadena de contexto enriquecida para el prefill del LLM
    pub fn build_augmented_prompt(
        &self,
        prompt: &str,
        query_vector: &[f32],
        max_tokens_context: usize,
    ) -> String {
        let matches = self.retrieve_context(query_vector, 2);
        if matches.is_empty() {
            return prompt.to_string();
        }

        let mut context_snippets = Vec::new();
        for m in matches {
            if m.similarity > 0.65 {
                let prefix = match m.niche {
                    IslandNiche::Episodic => "[Memoria Episódica]",
                    IslandNiche::Documental => "[Conocimiento Base]",
                    IslandNiche::Conversational => "[Historial Prevío]",
                };
                context_snippets.push(format!("{} {}", prefix, m.text));
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
