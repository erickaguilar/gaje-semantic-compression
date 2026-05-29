use crate::nn::llm::GenomicLLM;
use crate::core::tokenizer::GajeTokenizer;
use crate::io::loader::GGUFLoader;
use rayon::prelude::*;
use std::time::Instant;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Representa un maestro en el Consejo de Profesores.
#[cfg_attr(feature = "python", pyclass)]
#[derive(Clone)]
pub struct Teacher {
    pub name: String,
    pub model: GenomicLLM,
    pub tokenizer: GajeTokenizer,
    pub vocab_mapping: Vec<Option<usize>>, // teacher_token_id -> student_token_id
    pub is_identity_vocab: bool,
}

#[cfg_attr(feature = "python", pymethods)]
impl Teacher {
    /// Crea un nuevo maestro cargando un modelo GGUF y su tokenizador.
    #[cfg(feature = "python")]
    #[new]
    pub fn py_new(
        name: String, 
        model_path: &str, 
        tokenizer_path: &str, 
        student_tokenizer: &GajeTokenizer
    ) -> PyResult<Self> {
        Self::new(name, model_path, tokenizer_path, student_tokenizer).map_err(pyo3::exceptions::PyValueError::new_err)
    }
}

impl Teacher {
    /// Crea un nuevo maestro cargando un modelo GGUF y su tokenizador.
    /// El mapeo de vocabulario se pre-calcula comparando tokens decodificados.
    pub fn new(
        name: String, 
        model_path: &str, 
        tokenizer_path: &str, 
        student_tokenizer: &GajeTokenizer
    ) -> Result<Self, String> {
        println!("[*] Cargando Maestro '{}' desde {}...", name, model_path);
        let loader = GGUFLoader::new(model_path).map_err(|e| e.to_string())?;
        let config = loader.infer_config().map_err(|e| e.to_string())?;
        
        // Cargamos el maestro como un modelo genómico para ejecución nativa en Rust.
        let model = loader.load_genomic_llm(config, -1.0).map_err(|e| e.to_string())?;
        
        let tokenizer = GajeTokenizer::from_file(tokenizer_path).map_err(|e| e.to_string())?;
        
        let vocab_size = tokenizer.vocab_size();
        let mut vocab_mapping = vec![None; vocab_size];

        println!("[*] Pre-calculando mapeo de vocabulario para '{}' (Vocab: {})...", name, vocab_size);

        // Optimización: Si los tokenizadores son idénticos (basado en el tamaño del vocabulario 
        // y una prueba rápida de los primeros 100 tokens), podemos usar un mapeo de identidad.
        let mut is_identity = vocab_size == student_tokenizer.vocab_size();
        if is_identity {
            for i in 0..100.min(vocab_size) {
                if let Ok(t1) = tokenizer.decode(&[i as u32], true) {
                    if let Some(s_id) = student_tokenizer.token_to_id(&t1) {
                        if s_id as usize != i { is_identity = false; break; }
                    } else { is_identity = false; break; }
                }
            }
        }

        if is_identity {
            println!("    [+] Detectada identidad de vocabulario. Saltando mapeo exhaustivo.");
            for i in 0..vocab_size { vocab_mapping[i] = Some(i); }
        } else {
            // Paralelizar con Rayon para evitar bloqueos en dispositivos móviles
            use rayon::prelude::*;
            let results: Vec<Option<usize>> = (0..vocab_size).into_par_iter().map(|i| {
                if let Ok(token_str) = tokenizer.decode(&[i as u32], true) {
                    if !token_str.is_empty() {
                        return student_tokenizer.token_to_id(&token_str).map(|id| id as usize);
                    }
                }
                None
            }).collect();
            vocab_mapping = results;
            println!("    [✔] Mapeo completado (Rayon).");
        }

        Ok(Teacher { name, model, tokenizer, vocab_mapping, is_identity_vocab: is_identity })

    }
}

/// Orquestador de múltiples maestros para destilación por consenso.
#[cfg_attr(feature = "python", pyclass)]
#[derive(Clone)]
pub struct CouncilOfTeachers {
    pub teachers: Vec<Teacher>,
}

#[cfg_attr(feature = "python", pymethods)]
impl CouncilOfTeachers {
    #[cfg(feature = "python")]
    #[new]
    pub fn py_new() -> Self {
        Self::new()
    }

    #[cfg(feature = "python")]
    pub fn py_add_teacher(&mut self, teacher: Teacher) {
        self.add_teacher(teacher);
    }
}

impl CouncilOfTeachers {
    pub fn new() -> Self {
        CouncilOfTeachers { teachers: Vec::new() }
    }

    pub fn add_teacher(&mut self, teacher: Teacher) {
        self.teachers.push(teacher);
    }

    /// Obtiene las probabilidades del consejo para una secuencia de texto.
    pub fn get_consensus_probs(&self, text: &str, student_vocab_size: usize) -> Vec<Vec<f32>> {
        if self.teachers.is_empty() { return Vec::new(); }

        let teacher_results: Vec<Vec<Vec<f32>>> = self.teachers.par_iter().map(|teacher| {
            let mut model = teacher.model.clone();
            let mut tokens = match teacher.tokenizer.encode(text, false) {
                Ok(t) => t,
                Err(_) => return Vec::new(),
            };

            // Limitamos a 512 tokens para evitar bloqueos en Android con textos muy largos (Mosaic)
            if tokens.len() > 512 {
                tokens.truncate(512);
            }

            let mut seq_probs = Vec::with_capacity(tokens.len());
            model.clear_cache_core();

            for &token_id in tokens.iter() {
                // Feedback silencioso pero útil para depuración interna si fuera necesario
                match model.forward_core(token_id as usize, false) {
                    Ok(logits) => {
                        let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                        let mut sum_exp = 0.0f32;
                        let mut probs = vec![0.0f32; logits.len()];
                        for (i, &l) in logits.iter().enumerate() {
                            let e = (l - max_l).exp();
                            probs[i] = e;
                            sum_exp += e;
                        }
                        for p in &mut probs { *p /= sum_exp + 1e-12; }

                        if teacher.is_identity_vocab {
                            seq_probs.push(probs);
                        } else {
                            let mut student_probs = vec![0.0f32; student_vocab_size];
                            for (t_id, &p) in probs.iter().enumerate() {
                                if p > 1e-6 {
                                    if let Some(s_id) = teacher.vocab_mapping.get(t_id).and_then(|&id| id) {
                                        if let Some(slot) = student_probs.get_mut(s_id) {
                                            *slot += p;
                                        }
                                    }
                                }
                            }

                            let s: f32 = student_probs.iter().sum();
                            if s > 0.0 { for p in &mut student_probs { *p /= s; } }
                            seq_probs.push(student_probs);
                        }
                    },
                    Err(_) => {
                        seq_probs.push(vec![1.0 / student_vocab_size as f32; student_vocab_size]);
                    }
                }
            }
            seq_probs
        }).collect();

        let num_teachers = teacher_results.len();
        if num_teachers == 0 { return Vec::new(); }
        
        let seq_len = teacher_results[0].len();
        let mut consensus = Vec::with_capacity(seq_len);

        for i in 0..seq_len {
            let mut combined = vec![0.0f32; student_vocab_size];
            for t_idx in 0..num_teachers {
                if i < teacher_results[t_idx].len() {
                    let probs = &teacher_results[t_idx][i];
                    for v in 0..student_vocab_size {
                        combined[v] += probs[v];
                    }
                }
            }
            for v in 0..student_vocab_size {
                combined[v] /= num_teachers as f32;
            }
            consensus.push(combined);
        }

        consensus
    }
}

/// Implementa el ciclo de destilación nativa.
#[cfg_attr(feature = "python", pyclass)]
#[derive(Clone)]
pub struct NativeGenomicDistiller {
    pub inner: GenomicDistiller,
}

#[cfg_attr(feature = "python", pymethods)]
impl NativeGenomicDistiller {
    #[cfg(feature = "python")]
    #[new]
    #[pyo3(signature = (council, student_tokenizer, distill_weight=0.5))]
    pub fn py_new(council: CouncilOfTeachers, student_tokenizer: GajeTokenizer, distill_weight: f32) -> Self {
        let mut inner = GenomicDistiller::new(council, student_tokenizer);
        inner.distill_weight = distill_weight;
        NativeGenomicDistiller { inner }
    }

    #[cfg(feature = "python")]
    pub fn fit(
        &self,
        student: &mut GenomicLLM,
        texts: Vec<String>,
        epochs: usize,
        lr: f32,
    ) -> PyResult<()> {
        self.inner.fit(student, &texts, epochs, lr).map_err(pyo3::exceptions::PyValueError::new_err)
    }
}

#[derive(Clone)]
pub struct GenomicDistiller {
    pub council: CouncilOfTeachers,
    pub student_tokenizer: GajeTokenizer,
    pub distill_weight: f32, 
}

impl GenomicDistiller {
    pub fn new(council: CouncilOfTeachers, student_tokenizer: GajeTokenizer) -> Self {
        GenomicDistiller {
            council,
            student_tokenizer,
            distill_weight: 0.5,
        }
    }

    pub fn distill_step(
        &self,
        student: &mut GenomicLLM,
        text: &str,
        lr: f32,
    ) -> Result<f32, String> {
        let mut tokens = self.student_tokenizer.encode(text, false)
            .map_err(|e| format!("Error tokenizando estudiante: {}", e))?;
        
        if tokens.len() < 2 { return Ok(0.0); }
        
        // Truncar para consistencia con el profesor y velocidad en Android
        if tokens.len() > 512 { tokens.truncate(512); }

        let student_vocab_size = student.lm_head.out_features;
        let consensus_seq = self.council.get_consensus_probs(text, student_vocab_size);

        if consensus_seq.is_empty() { return Ok(0.0); }

        student.clear_cache_core();
        let mut total_loss = 0.0;
        let n_steps = (tokens.len() - 1).min(consensus_seq.len());
        
        // Feedback granular cada 50 tokens para que el usuario sepa que hay vida
        let show_progress = n_steps > 100;

        for i in 0..n_steps {
            if show_progress && i % 50 == 0 {
                print!("."); 
                use std::io::Write;
                std::io::stdout().flush().ok();
            }
            let token_id = tokens[i] as usize;
            let target_id = tokens[i+1] as usize;

            if token_id >= student_vocab_size || target_id >= student_vocab_size {
                continue;
            }

            let teacher_probs = &consensus_seq[i];

            let (logits, h_norm) = student.forward_with_hidden_core(token_id, false)?;
            
            let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum_exp = 0.0f32;
            let mut student_probs = vec![0.0f32; logits.len()];
            for (j, &l) in logits.iter().enumerate() {
                let e = (l - max_l).exp();
                student_probs[j] = e;
                sum_exp += e;
            }
            for p in &mut student_probs { *p /= sum_exp + 1e-12; }

            let ce_loss = -(student_probs[target_id].max(1e-10)).ln();
            
            let mut kl_loss = 0.0f32;
            for j in 0..logits.len() {
                let p_t = teacher_probs[j];
                let p_s = student_probs[j];
                if p_t > 1e-10 && p_s > 1e-10 {
                    kl_loss += p_t * (p_t.ln() - p_s.ln());
                }
            }

            let loss = (1.0 - self.distill_weight) * ce_loss + self.distill_weight * kl_loss;
            total_loss += loss;

            let mut d_logits = vec![0.0f32; logits.len()];
            for j in 0..logits.len() {
                let grad_ce = student_probs[j] - (if j == target_id { 1.0 } else { 0.0 });
                let grad_kl = student_probs[j] - teacher_probs[j];
                d_logits[j] = (1.0 - self.distill_weight) * grad_ce + self.distill_weight * grad_kl;
            }

            student.lm_head.refine_with_grads_core(h_norm, d_logits, lr)?;
        }

        Ok(total_loss / n_steps as f32)
    }

    pub fn fit(
        &self,
        student: &mut GenomicLLM,
        texts: &[String],
        epochs: usize,
        lr: f32,
    ) -> Result<(), String> {
        println!("[*] Iniciando Crianza por Destilación Nativa ({} épocas)", epochs);
        
        for epoch in 0..epochs {
            let start = Instant::now();
            let mut epoch_loss = 0.0;
            let mut count = 0;

            for text in texts {
                match self.distill_step(student, text, lr) {
                    Ok(loss) => {
                        epoch_loss += loss;
                        count += 1;
                        if count % 10 == 0 {
                            println!("    - Muestra {}/{} | Loss: {:.4} | {:?}", count, texts.len(), loss, start.elapsed());
                        }
                    },
                    Err(e) => println!("    [!] Error en muestra: {}", e),
                }
            }

            if count > 0 {
                let avg_loss = epoch_loss / count as f32;
                println!(
                    "✅ Época {}/{} completada | Loss Promedio: {:.4} | Tiempo: {:?}",
                    epoch + 1, epochs, avg_loss, start.elapsed()
                );
            }
        }
        Ok(())
    }
}
