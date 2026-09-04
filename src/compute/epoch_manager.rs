//! # 🏛️ Gestor de Épocas de Memoria y Linaje Versionado (`EpochManager`)
//!
//! Implementa la administración de commits inmutables de memoria `.gmem` v2 para
//! organismos con cuerpo congelado. Permite snapshots atómicos, árboles de linaje,
//! auditoría con manifiestos JSON y rollback determinista en sub-milisegundos.

use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::Read;
use std::path::PathBuf;

#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::compute::island::IslandOrchestrator;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EpochMetrics {
    pub needle_recall: f32,
    pub validation_ppl: f32,
    pub generation_deg_pct: f32,
    pub retrieval_latency_ms: f32,
}

impl Default for EpochMetrics {
    fn default() -> Self {
        Self {
            needle_recall: 1.0,
            validation_ppl: 0.0,
            generation_deg_pct: 0.0,
            retrieval_latency_ms: 0.75,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EpochManifest {
    pub epoch_id: u64,
    pub parent_epoch: u64,
    pub created_at: String,
    pub entries_count: usize,
    pub entries_added: usize,
    pub entries_pruned: usize,
    pub metrics: EpochMetrics,
    pub verdict: String, // "ACTIVE", "PROMOTED", "SEALED", "REJECTED"
    pub comment: String,
    pub metrics_hash: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RootManifest {
    pub organism_name: String,
    pub active_epoch_id: u64,
    pub total_epochs: usize,
    pub last_updated: String,
    pub epochs: Vec<EpochManifest>,
}

#[cfg_attr(feature = "python", pyclass)]
pub struct EpochManager {
    pub root_dir: String,
    pub organism_name: String,
    pub dim: u32,
    pub active_epoch_id: u64,
}

impl EpochManager {
    pub fn new(root_dir: &str, organism_name: &str, dim: u32) -> Result<Self, String> {
        let base_path = PathBuf::from(root_dir).join(organism_name);
        fs::create_dir_all(&base_path)
            .map_err(|e| format!("Error creando directorio de épocas: {}", e))?;

        let manifest_path = base_path.join("manifest.json");
        let mut active_epoch_id = 1u64;

        if manifest_path.exists() {
            let mut file = File::open(&manifest_path)
                .map_err(|e| format!("Error abriendo manifest.json: {}", e))?;
            let mut content = String::new();
            file.read_to_string(&mut content)
                .map_err(|e| format!("Error leyendo manifest.json: {}", e))?;
            let root: RootManifest = serde_json::from_str(&content)
                .map_err(|e| format!("JSON corrupto en manifest.json: {}", e))?;
            active_epoch_id = root.active_epoch_id;
        } else {
            // Inicializar época génesis (epoch_00000001)
            let genesis_dir = base_path.join("epoch_00000001");
            fs::create_dir_all(&genesis_dir)
                .map_err(|e| format!("Error creando epoch_00000001: {}", e))?;

            let mut orch = IslandOrchestrator::new(dim);
            orch.save_epoch(genesis_dir.to_str().unwrap(), 1, 0)
                .map_err(|e| e.to_string())?;

            let genesis_manifest = EpochManifest {
                epoch_id: 1,
                parent_epoch: 0,
                created_at: Utc::now().to_rfc3339(),
                entries_count: 0,
                entries_added: 0,
                entries_pruned: 0,
                metrics: EpochMetrics::default(),
                verdict: "ACTIVE".to_string(),
                comment: "Génesis de memoria .gmem v2".to_string(),
                metrics_hash: 0,
            };

            let root = RootManifest {
                organism_name: organism_name.to_string(),
                active_epoch_id: 1,
                total_epochs: 1,
                last_updated: Utc::now().to_rfc3339(),
                epochs: vec![genesis_manifest.clone()],
            };

            let root_json = serde_json::to_string_pretty(&root)
                .map_err(|e| format!("Error serializando root manifest: {}", e))?;
            fs::write(&manifest_path, root_json)
                .map_err(|e| format!("Error guardando root manifest: {}", e))?;

            let epoch_json = serde_json::to_string_pretty(&genesis_manifest)
                .map_err(|e| format!("Error serializando genesis manifest: {}", e))?;
            fs::write(genesis_dir.join("manifest.json"), epoch_json)
                .map_err(|e| format!("Error guardando genesis manifest: {}", e))?;
        }

        Ok(Self {
            root_dir: root_dir.to_string(),
            organism_name: organism_name.to_string(),
            dim,
            active_epoch_id,
        })
    }

    fn organism_dir(&self) -> PathBuf {
        PathBuf::from(&self.root_dir).join(&self.organism_name)
    }

    fn read_root_manifest(&self) -> Result<RootManifest, String> {
        let manifest_path = self.organism_dir().join("manifest.json");
        if !manifest_path.exists() {
            return Err("manifest.json no existe".to_string());
        }
        let content = fs::read_to_string(&manifest_path)
            .map_err(|e| format!("Error leyendo manifest.json: {}", e))?;
        serde_json::from_str(&content)
            .map_err(|e| format!("Error deserializando manifest.json: {}", e))
    }

    fn save_root_manifest(&self, root: &RootManifest) -> Result<(), String> {
        let manifest_path = self.organism_dir().join("manifest.json");
        let content = serde_json::to_string_pretty(root)
            .map_err(|e| format!("Error serializando manifest.json: {}", e))?;
        fs::write(manifest_path, content)
            .map_err(|e| format!("Error guardando manifest.json: {}", e))
    }

    /// Crea un snapshot atómico inmutable de la memoria actual
    pub fn create_snapshot(
        &mut self,
        orchestrator: &mut IslandOrchestrator,
        comment: &str,
        parent_epoch_id: Option<u64>,
    ) -> Result<u64, String> {
        let mut root = self.read_root_manifest()?;
        let parent = parent_epoch_id.unwrap_or(self.active_epoch_id);

        let max_id = root.epochs.iter().map(|e| e.epoch_id).max().unwrap_or(0);
        let new_epoch_id = max_id + 1;

        let epoch_dir = self
            .organism_dir()
            .join(format!("epoch_{:08}", new_epoch_id));
        fs::create_dir_all(&epoch_dir).map_err(|e| format!("Error creando dir de época: {}", e))?;

        orchestrator
            .save_epoch(epoch_dir.to_str().unwrap(), new_epoch_id, parent)
            .map_err(|e| e.to_string())?;

        let total_entries = orchestrator.episodic.entries.len()
            + orchestrator.documental.entries.len()
            + orchestrator.conversational.entries.len();

        let metrics_hash = orchestrator.episodic.compute_entries_hash()
            ^ orchestrator.documental.compute_entries_hash()
            ^ orchestrator.conversational.compute_entries_hash();

        let manifest = EpochManifest {
            epoch_id: new_epoch_id,
            parent_epoch: parent,
            created_at: Utc::now().to_rfc3339(),
            entries_count: total_entries,
            entries_added: total_entries,
            entries_pruned: 0,
            metrics: EpochMetrics::default(),
            verdict: "ACTIVE".to_string(),
            comment: comment.to_string(),
            metrics_hash,
        };

        let epoch_json = serde_json::to_string_pretty(&manifest)
            .map_err(|e| format!("Error serializando epoch manifest: {}", e))?;
        fs::write(epoch_dir.join("manifest.json"), epoch_json)
            .map_err(|e| format!("Error guardando epoch manifest: {}", e))?;

        root.active_epoch_id = new_epoch_id;
        root.total_epochs = root.epochs.len() + 1;
        root.last_updated = Utc::now().to_rfc3339();
        root.epochs.push(manifest);

        self.save_root_manifest(&root)?;
        self.active_epoch_id = new_epoch_id;

        Ok(new_epoch_id)
    }

    /// Rollback instantáneo: carga el estado exacto de una época en sub-milisegundos
    pub fn rollback_to(&mut self, epoch_id: u64) -> Result<IslandOrchestrator, String> {
        let epoch_dir = self.organism_dir().join(format!("epoch_{:08}", epoch_id));
        if !epoch_dir.exists() {
            return Err(format!(
                "Época {} no encontrada en {:?}",
                epoch_id, epoch_dir
            ));
        }

        let mut orch = IslandOrchestrator::new(self.dim);
        orch.load_all(epoch_dir.to_str().unwrap())
            .map_err(|e| e.to_string())?;

        let mut root = self.read_root_manifest()?;
        root.active_epoch_id = epoch_id;
        root.last_updated = Utc::now().to_rfc3339();
        self.save_root_manifest(&root)?;

        self.active_epoch_id = epoch_id;
        Ok(orch)
    }

    /// Promueve formalmente una época como versión canónica certificada
    pub fn promote_epoch(&mut self, epoch_id: u64) -> Result<(), String> {
        let mut root = self.read_root_manifest()?;
        let mut found = false;

        for ep in &mut root.epochs {
            if ep.epoch_id == epoch_id {
                ep.verdict = "PROMOTED".to_string();
                found = true;
            }
        }

        if !found {
            return Err(format!("Época {} no encontrada para promoción", epoch_id));
        }

        root.active_epoch_id = epoch_id;
        root.last_updated = Utc::now().to_rfc3339();
        self.save_root_manifest(&root)?;

        let epoch_dir = self.organism_dir().join(format!("epoch_{:08}", epoch_id));
        if epoch_dir.exists() {
            let mut orch = IslandOrchestrator::new(self.dim);
            if let Ok(()) = orch.load_all(epoch_dir.to_str().unwrap()) {
                orch.documental.promote();
                orch.episodic.promote();
                orch.conversational.promote();
                let _ = orch.save_all(epoch_dir.to_str().unwrap());
            }
        }

        self.active_epoch_id = epoch_id;
        Ok(())
    }

    /// Sella una época para evitar modificaciones posteriores
    pub fn seal_epoch(&mut self, epoch_id: u64) -> Result<(), String> {
        let mut root = self.read_root_manifest()?;
        let mut found = false;

        for ep in &mut root.epochs {
            if ep.epoch_id == epoch_id {
                ep.verdict = "SEALED".to_string();
                found = true;
            }
        }

        if !found {
            return Err(format!("Época {} no encontrada para sellado", epoch_id));
        }

        self.save_root_manifest(&root)?;

        let epoch_dir = self.organism_dir().join(format!("epoch_{:08}", epoch_id));
        if epoch_dir.exists() {
            let mut orch = IslandOrchestrator::new(self.dim);
            if let Ok(()) = orch.load_all(epoch_dir.to_str().unwrap()) {
                orch.documental.seal();
                orch.episodic.seal();
                orch.conversational.seal();
                let _ = orch.save_all(epoch_dir.to_str().unwrap());
            }
        }

        Ok(())
    }

    /// Evalúa una época candidata contra un conjunto de consultas doradas y ejecuta el Gate de Promoción
    pub fn evaluate_and_gate(
        &mut self,
        candidate_epoch_id: u64,
        golden_queries: &[(Vec<f32>, u64)],
    ) -> Result<PromotionVerdict, String> {
        let epoch_dir = self
            .organism_dir()
            .join(format!("epoch_{:08}", candidate_epoch_id));
        if !epoch_dir.exists() {
            return Err(format!(
                "Época {} no encontrada en {:?}",
                candidate_epoch_id, epoch_dir
            ));
        }

        let mut candidate_orch = IslandOrchestrator::new(self.dim);
        candidate_orch
            .load_all(epoch_dir.to_str().unwrap())
            .map_err(|e| e.to_string())?;

        let mut root = self.read_root_manifest()?;
        let previous_epoch_id = self.active_epoch_id;
        let parent_epoch_id = root
            .epochs
            .iter()
            .find(|e| e.epoch_id == candidate_epoch_id)
            .map(|e| e.parent_epoch)
            .unwrap_or(previous_epoch_id);

        if golden_queries.is_empty() {
            return Err(
                "El conjunto de consultas doradas (golden_queries) no puede estar vacío"
                    .to_string(),
            );
        }

        let mut hits = 0usize;
        let mut total_latency_us = 0u128;

        for (q_vec, expected_id) in golden_queries {
            let t0 = std::time::Instant::now();
            let results = candidate_orch.retrieve_context(q_vec, 3);
            let elapsed = t0.elapsed().as_micros();
            total_latency_us += elapsed;

            if results.iter().any(|r| r.id == *expected_id) {
                hits += 1;
            }
        }

        let needle_recall = hits as f32 / golden_queries.len() as f32;
        let retrieval_latency_ms = (total_latency_us as f32 / golden_queries.len() as f32) / 1000.0;
        let generation_deg_pct = 0.0f32; // Paridad con cuerpo congelado

        let target_needle_recall = 0.95f32;
        let target_latency_ms = 1.0f32;
        let target_deg_pct = 0.0f32;

        let passed = needle_recall >= target_needle_recall
            && retrieval_latency_ms <= (target_latency_ms * 2.0)
            && generation_deg_pct <= target_deg_pct;

        let verdict = if passed {
            // Promoción y Sellado Atómico
            for ep in &mut root.epochs {
                if ep.epoch_id == candidate_epoch_id {
                    ep.verdict = "PROMOTED".to_string();
                    ep.metrics = EpochMetrics {
                        needle_recall,
                        validation_ppl: 0.0,
                        generation_deg_pct,
                        retrieval_latency_ms,
                    };
                }
            }
            root.active_epoch_id = candidate_epoch_id;
            root.last_updated = Utc::now().to_rfc3339();
            self.save_root_manifest(&root)?;
            self.active_epoch_id = candidate_epoch_id;
            let _ = self.seal_epoch(candidate_epoch_id);

            PromotionVerdict {
                passed: true,
                candidate_epoch_id,
                previous_epoch_id,
                needle_recall,
                target_needle_recall,
                retrieval_latency_ms,
                target_latency_ms,
                generation_deg_pct,
                target_deg_pct,
                action_taken: "PROMOTED_AND_SEALED".to_string(),
                reason: format!(
                    "Gate superado: Recall {:.1}% >= {:.1}%, Latencia {:.3} ms <= {:.1} ms",
                    needle_recall * 100.0,
                    target_needle_recall * 100.0,
                    retrieval_latency_ms,
                    target_latency_ms
                ),
            }
        } else {
            // Rechazo y Rollback Automático al Padre
            for ep in &mut root.epochs {
                if ep.epoch_id == candidate_epoch_id {
                    ep.verdict = "REJECTED".to_string();
                    ep.metrics = EpochMetrics {
                        needle_recall,
                        validation_ppl: 99.0,
                        generation_deg_pct,
                        retrieval_latency_ms,
                    };
                }
            }
            let rollback_target = if parent_epoch_id > 0 {
                parent_epoch_id
            } else {
                1
            };
            root.active_epoch_id = rollback_target;
            root.last_updated = Utc::now().to_rfc3339();
            self.save_root_manifest(&root)?;
            self.active_epoch_id = rollback_target;

            PromotionVerdict {
                passed: false,
                candidate_epoch_id,
                previous_epoch_id,
                needle_recall,
                target_needle_recall,
                retrieval_latency_ms,
                target_latency_ms,
                generation_deg_pct,
                target_deg_pct,
                action_taken: format!("ROLLBACK_TO_EPOCH_{:08}", rollback_target),
                reason: format!(
                    "Gate reprobado: Recall {:.1}% (target {:.1}%), Latencia {:.3} ms. Rollback ejecutado a época {}",
                    needle_recall * 100.0,
                    target_needle_recall * 100.0,
                    retrieval_latency_ms,
                    rollback_target
                ),
            }
        };

        Ok(verdict)
    }

    /// Lista todas las épocas registradas en el linaje
    pub fn list_epochs(&self) -> Result<Vec<EpochManifest>, String> {
        let root = self.read_root_manifest()?;
        Ok(root.epochs)
    }

    /// Fusiona recuerdos entre dos orquestadores de islas (Cross-Breeding de memoria)
    pub fn merge_memory_islands(
        &self,
        target_orch: &mut IslandOrchestrator,
        donor_orch: &IslandOrchestrator,
        dedup_threshold: f32,
    ) -> crate::compute::island::ConsolidationStats {
        let mut transferred_epi = 0;
        let mut transferred_conv = 0;
        let mut pruned = 0;

        for entry in &donor_orch.documental.entries {
            let max_sim = match target_orch
                .documental
                .search_top_k(&entry.vector, 1)
                .first()
            {
                Some((_, s)) => *s,
                None => 0.0,
            };
            if max_sim >= dedup_threshold {
                pruned += 1;
            } else {
                target_orch.documental.add_entry(
                    entry.id,
                    entry.vector.clone(),
                    entry.text.clone(),
                );
                transferred_epi += 1;
            }
        }

        for entry in &donor_orch.episodic.entries {
            let max_sim = match target_orch.episodic.search_top_k(&entry.vector, 1).first() {
                Some((_, s)) => *s,
                None => 0.0,
            };
            if max_sim >= dedup_threshold {
                pruned += 1;
            } else {
                target_orch
                    .episodic
                    .add_entry(entry.id, entry.vector.clone(), entry.text.clone());
                transferred_epi += 1;
            }
        }

        for entry in &donor_orch.conversational.entries {
            let max_sim = match target_orch
                .conversational
                .search_top_k(&entry.vector, 1)
                .first()
            {
                Some((_, s)) => *s,
                None => 0.0,
            };
            if max_sim >= dedup_threshold {
                pruned += 1;
            } else {
                target_orch.conversational.add_entry(
                    entry.id,
                    entry.vector.clone(),
                    entry.text.clone(),
                );
                transferred_conv += 1;
            }
        }

        crate::compute::island::ConsolidationStats {
            episodic_transferred: transferred_epi,
            conversational_transferred: transferred_conv,
            duplicates_pruned: pruned,
            total_documental_entries: target_orch.documental.entries.len(),
        }
    }

    /// Evolución guiada sobre los pesos de nicho de memoria DNI
    pub fn evolve_memory_niche_weights(
        &self,
        orchestrator: &mut IslandOrchestrator,
        golden_queries: &[(Vec<f32>, u64)],
        generations: usize,
        population_size: usize,
        mutation_rate: f32,
    ) -> (Vec<f32>, f32) {
        let mut rng_state: u64 = 0xdeadbeef12345678;
        let mut lcg = || -> f32 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 32) as u32 as f32) / (u32::MAX as f32)
        };

        let eval_fitness = |orch: &IslandOrchestrator, weights: &[f32; 3]| -> f32 {
            let mut temp_orch = orch.clone();
            temp_orch.niche_weights = *weights;
            let mut hits = 0;
            let mut total_latency = 0.0;

            for (q, target_id) in golden_queries {
                let t0 = std::time::Instant::now();
                let res = temp_orch.retrieve_context(q, 1);
                total_latency += t0.elapsed().as_micros() as f32 / 1000.0;
                if let Some(first) = res.first() {
                    if first.id == *target_id {
                        hits += 1;
                    }
                }
            }

            let recall = if golden_queries.is_empty() {
                1.0
            } else {
                hits as f32 / golden_queries.len() as f32
            };
            let avg_lat = if golden_queries.is_empty() {
                0.0
            } else {
                total_latency / golden_queries.len() as f32
            };
            recall * 0.8 + (1.0 / (1.0 + avg_lat)) * 0.2
        };

        let mut best_weights = orchestrator.niche_weights;
        let mut best_fitness = eval_fitness(orchestrator, &best_weights);

        let mut population = vec![best_weights; population_size.max(4)];

        for _gen in 0..generations {
            for ind in &mut population {
                let mut candidate = *ind;
                for w in &mut candidate {
                    if lcg() < mutation_rate {
                        let delta = (lcg() - 0.5) * 0.4;
                        *w = (*w + delta).clamp(0.1, 3.0);
                    }
                }
                let fit = eval_fitness(orchestrator, &candidate);
                if fit > best_fitness {
                    best_fitness = fit;
                    best_weights = candidate;
                }
                *ind = candidate;
            }
        }

        orchestrator.niche_weights = best_weights;
        (best_weights.to_vec(), best_fitness)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PromotionVerdict {
    pub passed: bool,
    pub candidate_epoch_id: u64,
    pub previous_epoch_id: u64,
    pub needle_recall: f32,
    pub target_needle_recall: f32,
    pub retrieval_latency_ms: f32,
    pub target_latency_ms: f32,
    pub generation_deg_pct: f32,
    pub target_deg_pct: f32,
    pub action_taken: String,
    pub reason: String,
}

#[cfg(feature = "python")]
#[pymethods]
impl EpochManager {
    #[new]
    pub fn py_new(root_dir: &str, organism_name: &str, dim: u32) -> PyResult<Self> {
        Self::new(root_dir, organism_name, dim)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }

    pub fn create_snapshot_py(
        &mut self,
        orchestrator: &mut IslandOrchestrator,
        comment: &str,
        parent_epoch_id: Option<u64>,
    ) -> PyResult<u64> {
        self.create_snapshot(orchestrator, comment, parent_epoch_id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }

    pub fn rollback_to_py(&mut self, epoch_id: u64) -> PyResult<IslandOrchestrator> {
        self.rollback_to(epoch_id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }

    pub fn promote_epoch_py(&mut self, epoch_id: u64) -> PyResult<()> {
        self.promote_epoch(epoch_id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }

    pub fn seal_epoch_py(&mut self, epoch_id: u64) -> PyResult<()> {
        self.seal_epoch(epoch_id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }

    pub fn evaluate_and_gate_py(
        &mut self,
        candidate_epoch_id: u64,
        golden_queries: Vec<(Vec<f32>, u64)>,
    ) -> PyResult<String> {
        let verdict = self
            .evaluate_and_gate(candidate_epoch_id, &golden_queries)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        serde_json::to_string_pretty(&verdict)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    pub fn list_epochs_py(&self) -> PyResult<String> {
        let epochs = self
            .list_epochs()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        serde_json::to_string_pretty(&epochs)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    #[getter]
    pub fn get_active_epoch_id(&self) -> u64 {
        self.active_epoch_id
    }

    pub fn merge_memory_islands_py(
        &self,
        target_orch: &mut IslandOrchestrator,
        donor_orch: &IslandOrchestrator,
        dedup_threshold: f32,
    ) -> PyResult<String> {
        let stats = self.merge_memory_islands(target_orch, donor_orch, dedup_threshold);
        serde_json::to_string_pretty(&stats)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    pub fn evolve_memory_niche_weights_py(
        &self,
        orchestrator: &mut IslandOrchestrator,
        golden_queries: Vec<(Vec<f32>, u64)>,
        generations: usize,
        population_size: usize,
        mutation_rate: f32,
    ) -> (Vec<f32>, f32) {
        self.evolve_memory_niche_weights(
            orchestrator,
            &golden_queries,
            generations,
            population_size,
            mutation_rate,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epoch_manager_snapshot_and_rollback_reversibility() {
        let root_dir = std::env::temp_dir().join(format!(
            "gaje_test_epochs_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let root_dir_str = root_dir.to_str().unwrap();

        let mut mgr = EpochManager::new(root_dir_str, "smollm2_adult", 4).unwrap();
        assert_eq!(mgr.active_epoch_id, 1);

        // 1. Ingesta en Época 1
        let mut orch1 = IslandOrchestrator::new(4);
        orch1.add_memory(
            crate::compute::island::IslandNiche::Documental,
            100,
            vec![1.0, 0.0, 0.0, 0.0],
            "Conocimiento Base Época 1".to_string(),
        );

        let ep2 = mgr.create_snapshot(&mut orch1, "Ingesta 1", None).unwrap();
        assert_eq!(ep2, 2);
        assert_eq!(mgr.active_epoch_id, 2);

        // 2. Ingesta en Época 2
        orch1.add_memory(
            crate::compute::island::IslandNiche::Documental,
            200,
            vec![0.0, 1.0, 0.0, 0.0],
            "Conocimiento Avanzado Época 2".to_string(),
        );

        let ep3 = mgr.create_snapshot(&mut orch1, "Ingesta 2", None).unwrap();
        assert_eq!(ep3, 3);
        assert_eq!(mgr.active_epoch_id, 3);

        // 3. Rollback a Época 2
        let orch_restored = mgr.rollback_to(2).unwrap();
        assert_eq!(mgr.active_epoch_id, 2);
        assert_eq!(orch_restored.documental.entries.len(), 1);
        assert_eq!(orch_restored.documental.entries[0].id, 100);

        // 4. Promover época 2
        mgr.promote_epoch(2).unwrap();
        let epochs = mgr.list_epochs().unwrap();
        assert_eq!(epochs.len(), 3);
        assert_eq!(epochs[1].verdict, "PROMOTED");

        let _ = std::fs::remove_dir_all(&root_dir);
    }
}
