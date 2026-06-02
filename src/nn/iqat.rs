//! # 🧬 IQAT: Iterative Quantization-Aware Training
//! 
//! Este módulo implementa el refinamiento profundo del modelo genómico
//! mediante la minimización de la "Deriva de Activación" (Activation Drift)
//! respecto a un modelo maestro.

use crate::nn::llm::GenomicLLM;
use crate::nn::distiller::Teacher;
use crate::core::tokenizer::GajeTokenizer;
use std::time::Instant;

#[cfg(feature = "python")]
use pyo3::prelude::*;

pub struct IQATEngine {
    pub lr: f32,
    pub block_lr_scale: f32,
}

impl IQATEngine {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            block_lr_scale: 0.1, // El refinamiento interno es más sutil
        }
    }

    /// Refina un bloque específico del estudiante para que sus activaciones
    /// coincidan con las del maestro.
    pub fn refine_block_drift(
        &self,
        student_model: &mut GenomicLLM,
        block_idx: usize,
        input_tokens: &[usize],
        teacher: &Teacher,
    ) -> Result<f32, String> {
        let mut total_drift = 0.0;
        let mut count = 0;

        // Necesitamos capturar activaciones del maestro y del estudiante
        // en el punto de entrada y salida del bloque.
        
        let mut teacher_model = teacher.model.clone();
        
        for &token_id in input_tokens {
            // 1. Obtener activaciones del maestro hasta el bloque bloque_idx
            let t_act_in = self.capture_activation(&mut teacher_model, token_id, block_idx)?;
            let t_act_out = self.capture_activation(&mut teacher_model, token_id, block_idx + 1)?;

            // 2. Obtener activaciones del estudiante hasta el bloque bloque_idx
            let s_act_in = self.capture_activation(student_model, token_id, block_idx)?;
            
            // 3. Ejecutar bloque del estudiante con la entrada real del estudiante
            let block = &mut student_model.blocks[block_idx];
            let s_act_out = block.forward_core(s_act_in.clone(), count)?;

            // 4. Calcular Deriva (Drift) y Gradiente
            // L = 0.5 * |s_out - t_out|^2 => dL/ds_out = s_out - t_out
            let mut grads = vec![0.0f32; s_act_out.len()];
            let mut drift = 0.0f32;
            for j in 0..s_act_out.len() {
                let diff = s_act_out[j] - t_act_out[j];
                grads[j] = diff;
                drift += diff * diff;
            }
            
            total_drift += drift.sqrt();
            count += 1;

            // 5. Refinamiento quirúrgico del bloque (Backprop local)
            // Refinamos las sub-capas del bloque usando el gradiente de deriva
            block.gate_gen.refine_with_grads_core(s_act_in.clone(), grads.clone(), self.lr * self.block_lr_scale)?;
            block.up_gen.refine_with_grads_core(s_act_in.clone(), grads.clone(), self.lr * self.block_lr_scale)?;
            // Nota: ffn_down requiere su propia entrada (post-act)
        }

        Ok(total_drift / count as f32)
    }

    fn capture_activation(&self, model: &mut GenomicLLM, token_id: usize, block_limit: usize) -> Result<Vec<f32>, String> {
        // Ejecución parcial del modelo hasta block_limit
        let mut x = model.embeddings.get_row_core(token_id)?;
        for i in 0..block_limit {
            if i < model.blocks.len() {
                x = model.blocks[i].forward_core(x, 0)?; // pos=0 para simplificar
            }
        }
        Ok(x)
    }

    pub fn run_iqat_cycle(
        &self,
        student: &mut GenomicLLM,
        teacher: &Teacher,
        texts: &[String],
        epochs: usize,
    ) -> Result<(), String> {
        println!("[*] Iniciando Ciclo IQAT (Iterative QAT) Nativo");
        let n_blocks = student.blocks.len();

        for epoch in 0..epochs {
            let start = Instant::now();
            println!("  [+] Época {}/{}", epoch + 1, epochs);

            for (t_idx, text) in texts.iter().enumerate() {
                // Usamos el tokenizador del maestro para asegurar alineación si el estudiante no tiene uno
                let tokens_u32 = teacher.tokenizer.encode(text, false).map_err(|e| e.to_string())?;
                let tokens: Vec<usize> = tokens_u32.into_iter().map(|id| id as usize).collect();

                if tokens.len() < 2 { continue; }

                for b_idx in 0..n_blocks {
                    let drift = self.refine_block_drift(student, b_idx, &tokens, teacher)?;
                    if t_idx % 10 == 0 && b_idx == n_blocks - 1 {
                        println!("    - Texto {} | Bloque {} | Drift Medio: {:.6}", t_idx, b_idx, drift);
                    }
                }
            }
            println!("  [✔] Época completada en {:?}", start.elapsed());
        }
        Ok(())
    }
}

#[cfg(feature = "python")]
#[pyclass]
pub struct NativeIQATEngine {
    inner: IQATEngine,
}

#[cfg(feature = "python")]
#[pymethods]
impl NativeIQATEngine {
    #[new]
    #[pyo3(signature = (lr=0.001))]
    pub fn new(lr: f32) -> Self {
        NativeIQATEngine {
            inner: IQATEngine::new(lr),
        }
    }

    pub fn run_cycle(
        &self,
        student: &mut GenomicLLM,
        teacher: &Teacher,
        texts: Vec<String>,
        epochs: usize,
    ) -> PyResult<()> {
        self.inner.run_iqat_cycle(student, teacher, &texts, epochs).map_err(pyo3::exceptions::PyValueError::new_err)
    }
}
