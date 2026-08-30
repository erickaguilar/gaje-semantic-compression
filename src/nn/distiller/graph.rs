// =============================================================================
// graph — DistillationGraph: grafo maestro-alumno para DNI en línea
// =============================================================================
//
// Escalado del CouncilOfTeachers estrella (N→1) a grafo completo N→M:
//   Nodos = Teacher(Qwen 3B, Pico 135M, ...) + Student(max.gaje Q2_0, ...)
//   Aristas = (teacher, student, alpha, temperature, weight)
//
// Ej.: pro_3b(0.6)+coder_3b(0.4) → max.gaje
//      pro_3b(0.3)+coder_3b(0.7) → max_code.gaje
// Todo batch 32 en VRAM vía GpuOnlineDistiller (kl + ste) zero-copy.
//
// Se apoya en StateGraph/ToT existente (src/compute/graph.rs) pero sin
// depender de su trait, para mantener pipeline.rs GPU puro.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::core::tokenizer::GajeTokenizer;
use crate::nn::distiller::Teacher;
use crate::nn::llm::GenomicLLM;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

#[derive(Debug, Clone, Copy)]
pub enum NodeKind { Teacher, Student }

#[derive(Debug, Clone)]
pub struct DistillNode {
    pub id: NodeId,
    pub kind: NodeKind,
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct DistillEdge {
    pub teacher: NodeId,
    pub student: NodeId,
    pub alpha: f32,       // peso KL (0..1)
    pub temperature: f32, // suavizado softmax
    pub weight: f32,      // ponderación del teacher en consenso
}

/// Grafo de destilación maestro→alumno.
/// Teachers y Students se registran como nodos; las aristas definen
/// qué teacher enseña a qué student con qué hiperparámetros.
/// La ejecución batch 32 usa `GpuOnlineDistiller` por arista cuando
/// hay GPU, fallback a `GenomicDistiller::distill_step` CPU.
pub struct DistillationGraph {
    nodes: HashMap<NodeId, DistillNode>,
    teachers: HashMap<NodeId, Teacher>,
    students: HashMap<NodeId, Arc<Mutex<GenomicLLM>>>,
    student_tokenizers: HashMap<NodeId, GajeTokenizer>,
    edges: Vec<DistillEdge>,
    next_id: usize,
}

impl DistillationGraph {
    pub fn new() -> Self {
        Self { nodes: HashMap::new(), teachers: HashMap::new(), students: HashMap::new(), student_tokenizers: HashMap::new(), edges: Vec::new(), next_id: 0 }
    }

    fn alloc_id(&mut self) -> NodeId { let id = NodeId(self.next_id); self.next_id += 1; id }

    /// Registra un Teacher (modelo + tokenizer) como nodo.
    pub fn add_teacher(&mut self, teacher: Teacher) -> NodeId {
        let id = self.alloc_id();
        let name = teacher.name.clone();
        self.nodes.insert(id, DistillNode { id, kind: NodeKind::Teacher, name: name.clone() });
        self.teachers.insert(id, teacher);
        id
    }

    /// Registra un Student Q2_0 (GenomicLLM + tokenizer) como nodo.
    pub fn add_student(&mut self, name: String, model: GenomicLLM, tok: GajeTokenizer) -> NodeId {
        let id = self.alloc_id();
        self.nodes.insert(id, DistillNode { id, kind: NodeKind::Student, name: name.clone() });
        self.students.insert(id, Arc::new(Mutex::new(model)));
        self.student_tokenizers.insert(id, tok);
        id
    }

    /// Crea arista maestro→alumno con hiperparámetros.
    pub fn add_edge(&mut self, teacher: NodeId, student: NodeId, alpha: f32, temperature: f32, weight: f32) -> Result<(), String> {
        if !self.teachers.contains_key(&teacher) { return Err(format!("teacher {:?} no existe", teacher)); }
        if !self.students.contains_key(&student) { return Err(format!("student {:?} no existe", student)); }
        if !(0.0..=1.0).contains(&alpha) { return Err("alpha 0..1".into()); }
        self.edges.push(DistillEdge { teacher, student, alpha, temperature, weight });
        Ok(())
    }

    /// Consenso batch 32 para un student: mezcla ponderada de teachers conectados.
    /// Usa GPU batch si disponible (GpuOnlineDistiller), si no CPU.
    pub fn consensus_for_student(&self, student: NodeId, text: &str) -> Vec<Vec<f32>> {
        let edges: Vec<&DistillEdge> = self.edges.iter().filter(|e| e.student == student).collect();
        if edges.is_empty() { return Vec::new(); }
        // Si hay un solo teacher, reusar council path rápido
        // Si múltiples, promediar ponderado por weight
        let vocab = self.students.get(&student).map(|s| s.lock().unwrap().lm_head.out_features).unwrap_or(49152);
        let mut acc: Option<Vec<Vec<f32>>> = None;
        let mut sum_w = 0.0f32;
        for e in edges {
            if let Some(t) = self.teachers.get(&e.teacher) {
                // council de un solo teacher para este edge
                let council = {
                    let mut c = crate::nn::distiller::CouncilOfTeachers::new();
                    c.add_teacher(t.clone());
                    c
                };
                let seq = council.get_consensus_probs(text, vocab);
                if seq.is_empty() { continue; }
                sum_w += e.weight;
                match &mut acc {
                    None => {
                        let mut weighted = seq;
                        for step in &mut weighted { for v in step.iter_mut() { *v *= e.weight; } }
                        acc = Some(weighted);
                    }
                    Some(cur) => {
                        for (i, step) in seq.iter().enumerate() {
                            if i < cur.len() { for (j, &p) in step.iter().enumerate() { cur[i][j] += p * e.weight; } }
                        }
                    }
                }
            }
        }
        if let Some(mut a) = acc {
            if sum_w > 0.0 { for step in &mut a { for v in step.iter_mut() { *v /= sum_w; } } }
            a
        } else { Vec::new() }
    }

    /// Paso de destilación para una arista (teacher→student) con texto.
    /// Intenta GPU zero-copy (batch 32) y cae a CPU.
    pub fn distill_edge(&self, edge: &DistillEdge, text: &str, lr: f32) -> Result<f32, String> {
        let student_arc = self.students.get(&edge.student).ok_or("student no encontrado")?.clone();
        let mut student = student_arc.lock().unwrap();
        let tok = self.student_tokenizers.get(&edge.student).ok_or("tokenizer no encontrado")?.clone();
        let consensus = self.consensus_for_student(edge.student, text);
        if consensus.is_empty() { return Ok(0.0); }
        // Path GPU si student es Q2_0 y hay GPU
        #[cfg(feature = "gpu")]
        {
            if let Some(d) = crate::compute::gpu::pipeline::GpuOnlineDistiller::try_new_global(32, edge.temperature, edge.alpha) {
                let tokens = tok.encode(text, false).map_err(|e| e.to_string())?;
                if tokens.len() >= 2 && !consensus.is_empty() {
                    let vocab = student.lm_head.out_features;
                    let batch = tokens.len().min(32).min(consensus.len());
                    let mut teacher_batch = Vec::with_capacity(batch * vocab);
                    let mut student_batch = Vec::with_capacity(batch * vocab);
                    student.clear_cache_core();
                    for i in 0..batch {
                        let tid = tokens[i] as usize;
                        let (logits, _) = student.forward_with_hidden_core(tid, false)?;
                        let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                        let sum_exp: f32 = logits.iter().map(|l| (l - max_l).exp()).sum();
                        for &l in &logits { student_batch.push(((l - max_l).exp())/(sum_exp+1e-12)); }
                        for &p in &consensus[i] { teacher_batch.push(p); }
                    }
                    let rows = student.lm_head.out_features;
                        let cols = student.lm_head.in_features;
                        if let Some(q2_db) = match &mut student.lm_head.weight_db {
                        crate::nn::linear::WeightDatabase::GenomicQ2_0(db) => Some(db),
                        _ => None,
                    } {
                        let db_mut: &mut Vec<crate::io::header::blocks::Q2_0Block> = std::sync::Arc::make_mut(q2_db);
                        if d.distill_step_online(&teacher_batch, &student_batch, db_mut, lr, rows, cols).is_ok() {
                            return Ok(0.0);
                        }
                    }
                }
            }
        }
        // Fallback CPU: usar GenomicDistiller por edge
        let mut council = crate::nn::distiller::CouncilOfTeachers::new();
        if let Some(t) = self.teachers.get(&edge.teacher) { council.add_teacher(t.clone()); }
        let distiller = crate::nn::distiller::GenomicDistiller::new(council, tok);
        // distiller usa self.distill_weight = alpha del edge
        let mut d2 = distiller;
        d2.distill_weight = edge.alpha;
        d2.distill_step(&mut student, text, lr)
    }

    /// Entrena todo el grafo sobre corpus, batch 32 por arista.
    pub fn fit_graph(&self, texts: &[String], epochs: usize, lr: f32) -> Result<(), String> {
        for epoch in 0..epochs {
            println!("🌐 Graph Epoch {}/{} — {} aristas, {} textos", epoch+1, epochs, self.edges.len(), texts.len());
            for edge in &self.edges {
                let mut edge_loss = 0.0;
                let mut cnt = 0;
                for txt in texts {
                    match self.distill_edge(edge, txt, lr) {
                        Ok(l) => { edge_loss += l; cnt += 1; },
                        Err(e) => eprintln!("  edge {:?}->{:?} err: {}", edge.teacher, edge.student, e),
                    }
                }
                if cnt>0 { println!("  edge {:?}->{:?} avg loss {:.4} (alpha={} temp={} w={})", edge.teacher, edge.student, edge_loss/cnt as f32, edge.alpha, edge.temperature, edge.weight); }
            }
        }
        Ok(())
    }

    pub fn describe(&self) {
        println!("DistillationGraph: {} nodos ({} teachers, {} students), {} aristas",
            self.nodes.len(), self.teachers.len(), self.students.len(), self.edges.len());
        for n in self.nodes.values() { println!("  {:?} {} {:?}", n.id, n.name, n.kind); }
        for e in &self.edges { println!("  {:?}->{:?} α={} T={} w={}", e.teacher, e.student, e.alpha, e.temperature, e.weight); }
    }
}

impl Default for DistillationGraph { fn default() -> Self { Self::new() } }
