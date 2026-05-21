use rand::Rng;
use std::time::Instant;
use rayon::prelude::*;
use std::fs;

#[derive(Clone)]
struct RecurrentMicroOrganism {
    dna_ih: Vec<u8>,
    dna_hh: Vec<u8>,
    dna_ho: Vec<u8>,
    centroids: [f32; 4],
    in_dim: usize,
    hidden_dim: usize,
    out_dim: usize,
}

impl RecurrentMicroOrganism {
    fn new(in_dim: usize, hidden_dim: usize, out_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let centroids = [-1.5, -0.5, 0.5, 1.5];
        let mut init_dna = |rows: usize, cols: usize| {
            let n_bytes = (rows * cols + 3) / 4;
            let mut dna = vec![0u8; n_bytes];
            rng.fill(&mut dna[..]);
            dna
        };
        Self {
            dna_ih: init_dna(hidden_dim, in_dim),
            dna_hh: init_dna(hidden_dim, hidden_dim),
            dna_ho: init_dna(out_dim, hidden_dim),
            centroids,
            in_dim,
            hidden_dim,
            out_dim,
        }
    }

    fn matmul_2bit(&self, dna: &[u8], input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; rows];
        let stride = cols / 4;
        for i in 0..rows {
            let mut sum = 0.0f32;
            let row_start = i * stride;
            for j in 0..stride {
                let byte = dna[row_start + j];
                for k in 0..4 {
                    let bits = (byte >> (k * 2)) & 0b11;
                    sum += input[j * 4 + k] * self.centroids[bits as usize];
                }
            }
            output[i] = sum;
        }
        output
    }

    fn step(&self, input: &[f32], hidden: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let ih = self.matmul_2bit(&self.dna_ih, input, self.hidden_dim, self.in_dim);
        let hh = self.matmul_2bit(&self.dna_hh, hidden, self.hidden_dim, self.hidden_dim);
        let mut new_hidden = vec![0.0f32; self.hidden_dim];
        for i in 0..self.hidden_dim {
            new_hidden[i] = (ih[i] + hh[i]).tanh();
        }
        let output = self.matmul_2bit(&self.dna_ho, &new_hidden, self.out_dim, self.hidden_dim);
        (output, new_hidden)
    }

    fn mutate(&mut self, n_mutations: usize) {
        let mut rng = rand::thread_rng();
        for _ in 0..n_mutations {
            let target_dna = match rng.gen_range(0..3) {
                0 => &mut self.dna_ih,
                1 => &mut self.dna_hh,
                _ => &mut self.dna_ho,
            };
            let byte_idx = rng.gen_range(0..target_dna.len());
            let bit_shift = rng.gen_range(0..4) * 2;
            let new_base = rng.gen_range(0..4) as u8;
            target_dna[byte_idx] &= !(0b11 << bit_shift);
            target_dna[byte_idx] |= new_base << bit_shift;
        }
    }
}

fn softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f32> = x.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|v| v / sum).collect()
}

fn main() {
    let dataset_path = "data/datasets/dataset_entrenamiento.txt";
    let target_text = fs::read_to_string(dataset_path).expect("No se pudo leer el dataset");
    let chars: Vec<char> = target_text.chars().collect();
    let mut vocab: Vec<char> = chars.clone();
    vocab.sort();
    vocab.dedup();
    let char_to_idx = |c: char| vocab.iter().position(|&x| x == c).unwrap();

    println!("🧬 Iniciando Crianza Evolutiva con Dataset Conversacional");
    println!("   Longitud del texto: {} caracteres", chars.len());
    println!("   Vocabulario: {} tokens únicos", vocab.len());

    let in_dim = vocab.len();
    let hidden_dim = 64; // Aumentado para mayor capacidad de memoria
    let out_dim = vocab.len();
    let mut organism = RecurrentMicroOrganism::new(in_dim, hidden_dim, out_dim);

    let mut best_total_fitness = 0.0f32;
    let iterations = 5000; // Reducido para una prueba rápida
    let population_size = 50;
    let start_time = Instant::now();

    for gen in 0..iterations {
        let paths: Vec<RecurrentMicroOrganism> = (0..population_size)
            .map(|_| {
                let mut clone = organism.clone();
                clone.mutate(5);
                clone
            })
            .collect();

        let results: Vec<(f32, RecurrentMicroOrganism)> = paths
            .into_par_iter()
            .map(|path| {
                let mut log_prob = 0.0f32;
                let mut current_hidden = vec![0.0f32; hidden_dim];
                // Evaluamos una sub-ventana para acelerar la evolución inicial
                let eval_len = std::cmp::min(chars.len(), 100); 
                for i in 0..eval_len - 1 {
                    let mut input = vec![0.0f32; in_dim];
                    input[char_to_idx(chars[i])] = 1.0;
                    let (output, next_hidden) = path.step(&input, &current_hidden);
                    let probs = softmax(&output);
                    log_prob += (probs[char_to_idx(chars[i+1])] + 1e-10).ln();
                    current_hidden = next_hidden;
                }
                (log_prob, path)
            })
            .collect();

        if let Some((max_log_prob, best_org)) = results.into_iter().max_by(|a, b| a.0.partial_cmp(&b.0).unwrap()) {
            if gen == 0 || max_log_prob > best_total_fitness {
                best_total_fitness = max_log_prob;
                organism = best_org;
                if gen % 100 == 0 {
                    println!("[Gen {}] Log-Probabilidad: {:.4} | Tiempo: {:?}", gen, best_total_fitness, start_time.elapsed());
                }
            }
        }
    }

    println!("\n✅ Crianza completada en {:?}", start_time.elapsed());
    println!("--- MUESTRA DE MEMORIA (Primeros 100 caracteres) ---");
    let mut current_hidden = vec![0.0f32; hidden_dim];
    print!("{}", chars[0]);
    for i in 0..std::cmp::min(chars.len(), 100) - 1 {
        let mut input = vec![0.0f32; in_dim];
        input[char_to_idx(chars[i])] = 1.0;
        let (output, next_hidden) = organism.step(&input, &current_hidden);
        let probs = softmax(&output);
        let best_idx = probs.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        print!("{}", vocab[best_idx]);
        current_hidden = next_hidden;
    }
    println!("\n---------------------------------------------------");
}
