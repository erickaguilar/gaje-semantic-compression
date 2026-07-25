use crate::nn::linear::GenomicLinear;
use crate::nn::llm::GenomicLLM;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Fusiona múltiples modelos genómicos promediando sus centroides en el espacio de fase.
/// Esto permite la integración de "Islas" de conocimiento sin destruir la cuantización base.
pub fn merge_genomic_models(models: &[GenomicLLM]) -> Result<GenomicLLM, String> {
    if models.is_empty() {
        return Err("No models to merge".to_string());
    }

    if models.len() == 1 {
        return Ok(models[0].clone());
    }

    let mut merged = models[0].clone();
    let n = models.len() as f32;

    println!(
        "[*] Fusionando {} organismos genómicos mediante promediado de centroides...",
        models.len()
    );

    // Helper para promediar centroides de una capa lineal
    let average_centroids = |target: &mut GenomicLinear, sources: &[&GenomicLinear]| {
        let n_centroids = target.centroids.len();
        for i in 0..n_centroids {
            let mut sum = 0.0;
            for s in sources {
                sum += s.centroids[i];
            }
            target.centroids[i] = sum / n;
        }
    };

    // Promediar Embeddings
    let emb_sources: Vec<&GenomicLinear> = models.iter().map(|m| &m.embeddings).collect();
    average_centroids(&mut merged.embeddings, &emb_sources);

    // Promediar LM Head
    let head_sources: Vec<&GenomicLinear> = models.iter().map(|m| &m.lm_head).collect();
    average_centroids(&mut merged.lm_head, &head_sources);

    // Promediar Bloques
    for b_idx in 0..merged.blocks.len() {
        let target_block = &mut merged.blocks[b_idx];

        let q_sources: Vec<&GenomicLinear> =
            models.iter().map(|m| &m.blocks[b_idx].q_gen).collect();
        average_centroids(&mut target_block.q_gen, &q_sources);

        let k_sources: Vec<&GenomicLinear> =
            models.iter().map(|m| &m.blocks[b_idx].k_gen).collect();
        average_centroids(&mut target_block.k_gen, &k_sources);

        let v_sources: Vec<&GenomicLinear> =
            models.iter().map(|m| &m.blocks[b_idx].v_gen).collect();
        average_centroids(&mut target_block.v_gen, &v_sources);

        let o_sources: Vec<&GenomicLinear> = models.iter().map(|m| &m.blocks[b_idx].w_o).collect();
        average_centroids(&mut target_block.w_o, &o_sources);

        let gate_sources: Vec<&GenomicLinear> =
            models.iter().map(|m| &m.blocks[b_idx].gate_gen).collect();
        average_centroids(&mut target_block.gate_gen, &gate_sources);

        let up_sources: Vec<&GenomicLinear> =
            models.iter().map(|m| &m.blocks[b_idx].up_gen).collect();
        average_centroids(&mut target_block.up_gen, &up_sources);

        let down_sources: Vec<&GenomicLinear> =
            models.iter().map(|m| &m.blocks[b_idx].w_down).collect();
        average_centroids(&mut target_block.w_down, &down_sources);
    }

    println!("[+] Fusión completada exitosamente.");
    Ok(merged)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "merge_models")]
pub fn merge_models_py(models: Vec<GenomicLLM>) -> PyResult<GenomicLLM> {
    merge_genomic_models(&models).map_err(pyo3::exceptions::PyValueError::new_err)
}
