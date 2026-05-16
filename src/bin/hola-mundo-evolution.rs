use rand::Rng;
use std::time::Instant;

/// Representa una base de ADN digital (2 bits)
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(u8)]
enum Base {
    A = 0b00,
    C = 0b01,
    G = 0b10,
    T = 0b11,
}

struct RecurrentMicroOrganism {
    dna_ih: Vec<u8>, // Pesos Entrada -> Oculto (2-bit)
    dna_hh: Vec<u8>, // Pesos Oculto -> Oculto (2-bit)
    dna_ho: Vec<u8>, // Pesos Oculto -> Salida (2-bit)
    centroids: [f32; 4],
    in_dim: usize,
    hidden_dim: usize,
    out_dim: usize,
}

impl RecurrentMicroOrganism {
    fn new(in_dim: usize, hidden_dim: usize, out_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let centroids = [-1.5, -0.5, 0.5, 1.5];
        
        let mut init_dna = |rows: usize, cols: usize| -> Vec<u8> {
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
        // h_t = tanh(W_ih * x_t + W_hh * h_{t-1})
        let ih = self.matmul_2bit(&self.dna_ih, input, self.hidden_dim, self.in_dim);
        let hh = self.matmul_2bit(&self.dna_hh, hidden, self.hidden_dim, self.hidden_dim);
        
        let mut new_hidden = vec![0.0f32; self.hidden_dim];
        for i in 0..self.hidden_dim {
            new_hidden[i] = (ih[i] + hh[i]).tanh();
        }
        
        // y_t = W_ho * h_t
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
    let target_text = "hola mundo";
    let chars: Vec<char> = target_text.chars().collect();
    let mut vocab: Vec<char> = chars.clone();
    vocab.sort();
    vocab.dedup();
    
    let char_to_idx = |c: char| vocab.iter().position(|&x| x == c).unwrap();
    
    println!("🧬 Evolucionando Memoria Secuencial: '{}'", target_text);
    println!("   Vocabulario: {:?}", vocab);

    let in_dim = vocab.len();
    let hidden_dim = 32;
    let out_dim = vocab.len();
    let mut organism = RecurrentMicroOrganism::new(in_dim, hidden_dim, out_dim);

    let mut best_total_fitness = 0.0f32;
    let iterations = 100_000;
    let start_time = Instant::now();

    for gen in 0..iterations {
        let old_ih = organism.dna_ih.clone();
        let old_hh = organism.dna_hh.clone();
        let old_ho = organism.dna_ho.clone();

        organism.mutate(3);

        let mut total_prob = 1.0f32;
        let mut current_hidden = vec![0.0f32; hidden_dim];
        
        // Simular secuencia
        for i in 0..chars.len() - 1 {
            let mut input = vec![0.0f32; in_dim];
            input[char_to_idx(chars[i])] = 1.0;
            
            let (output, next_hidden) = organism.step(&input, &current_hidden);
            let probs = softmax(&output);
            
            let target_char = chars[i+1];
            total_prob *= probs[char_to_idx(target_char)];
            current_hidden = next_hidden;
        }

        if total_prob > best_total_fitness {
            best_total_fitness = total_prob;
            if gen % 10000 == 0 || best_total_fitness > 0.5 {
                println!("[Gen {}] Probabilidad de Secuencia: {:.6}", gen, best_total_fitness);
            }
        } else {
            organism.dna_ih = old_ih;
            organism.dna_hh = old_hh;
            organism.dna_ho = old_ho;
        }

        if best_total_fitness > 0.95 {
            println!("🔥 ¡Evolución Exitosa! El organismo domina la secuencia.");
            break;
        }
    }

    println!("\n✅ Completado en {:?}", start_time.elapsed());
    
    // Verificación Final
    println!("--- PRUEBA DE MEMORIA ---");
    let mut current_hidden = vec![0.0f32; hidden_dim];
    print!("{}", chars[0]);
    for i in 0..chars.len() - 1 {
        let mut input = vec![0.0f32; in_dim];
        input[char_to_idx(chars[i])] = 1.0;
        let (output, next_hidden) = organism.step(&input, &current_hidden);
        let probs = softmax(&output);
        let best_idx = probs.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        print!("{}", vocab[best_idx]);
        current_hidden = next_hidden;
    }
    println!("\n-------------------------");
}
