use rand::Rng;
use std::time::Instant;
use rayon::prelude::*;

/// Representa una base de ADN digital (2 bits)
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(u8)]
#[allow(dead_code)]
enum Base {
    A = 0b00,
    C = 0b01,
    G = 0b10,
    T = 0b11,
}

#[allow(dead_code)]
impl Base {
    fn from_u8(v: u8) -> Self {
        match v % 4 {
            0 => Base::A,
            1 => Base::C,
            2 => Base::G,
            _ => Base::T,
        }
    }
}

#[derive(Clone)]
struct MicroOrganism {
    dna: Vec<u8>, // Cada u8 guarda 4 bases (2 bits c/u)
    centroids: [f32; 4],
    in_dim: usize,
    out_dim: usize,
}

impl MicroOrganism {
    fn new(in_dim: usize, out_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let n_bases = in_dim * out_dim;
        let n_bytes = (n_bases + 3) / 4;
        let mut dna = vec![0u8; n_bytes];
        rng.fill(&mut dna[..]);

        Self {
            dna,
            centroids: [-1.5, -0.5, 0.5, 1.5], // Centroides iniciales fijos
            in_dim,
            out_dim,
        }
    }

    /// Forward pass simplificado (MatMul en 2 bits)
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0f32; self.out_dim];
        let stride = self.in_dim / 4;

        for i in 0..self.out_dim {
            let mut sum = 0.0f32;
            let row_start = i * stride;
            for j in 0..stride {
                let byte = self.dna[row_start + j];
                // Desempaquetar 4 bases y multiplicar
                for k in 0..4 {
                    let bits = (byte >> (k * 2)) & 0b11;
                    let val = self.centroids[bits as usize];
                    sum += input[j * 4 + k] * val;
                }
            }
            output[i] = sum;
        }
        output
    }

    /// Mutación Monte Carlo: Cambia n bases aleatoriamente
    fn mutate(&mut self, n_mutations: usize) {
        let mut rng = rand::thread_rng();
        let n_bytes = self.dna.len();
        for _ in 0..n_mutations {
            let byte_idx = rng.gen_range(0..n_bytes);
            let bit_shift = rng.gen_range(0..4) * 2;
            let new_base = rng.gen_range(0..4) as u8;
            
            // Limpiar bits viejos y poner los nuevos
            self.dna[byte_idx] &= !(0b11 << bit_shift);
            self.dna[byte_idx] |= new_base << bit_shift;
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
    println!("🧬 Iniciando Evolución del Micro-Organismo GAJE...");
    
    // Configuración: 128 entradas -> 4 salidas (Logits para H, o, l, a)
    let in_dim = 128;
    let out_dim = 4;
    let mut organism = MicroOrganism::new(in_dim, out_dim);
    
    // El "Estimulo" de entrada (Trigger)
    let input = vec![1.0f32; in_dim];
    
    // El "Objetivo" (Deseamos que la salida máxima sea 'H' en la primera iteración, etc.)
    // Para simplificar, buscaremos que el primer logit sea el mayor (Representando coherencia)
    let target_idx = 0; // Queremos que el organismo aprenda a activar el canal 0
    
    let iterations = 20000;
    let mut best_fitness = f32::NEG_INFINITY;
    let start_time = Instant::now();
    let population_size = 100;

    for gen in 0..iterations {
        // Generar múltiples "caminos" paralelos
        let mut paths: Vec<MicroOrganism> = vec![];
        for _ in 0..population_size {
            let mut clone = organism.clone();
            clone.mutate(2); // 2 mutaciones por generación
            paths.push(clone);
        }

        // Evaluar la población en paralelo usando todos los núcleos (AVX2 implícito)
        let results: Vec<(f32, MicroOrganism)> = paths.into_par_iter().map(|path| {
            let logits = path.forward(&input);
            let probs = softmax(&logits);
            let fitness = probs[target_idx]; // Nuestra función de aptitud es la probabilidad del objetivo
            (fitness, path)
        }).collect();

        // Encontrar el camino más exitoso (Selección Natural)
        let mut best_path_fitness = f32::NEG_INFINITY;
        let mut best_path_organism = None;
        for (fitness, org) in results {
            if fitness > best_path_fitness {
                best_path_fitness = fitness;
                best_path_organism = Some(org);
            }
        }

        // Si el mejor mutante supera al ancestro, evoluciona
        if best_path_fitness > best_fitness {
            best_fitness = best_path_fitness;
            organism = best_path_organism.unwrap();
            println!("[Gen {}] Mejor Fitness (Probabilidad): {:.4}", gen, best_fitness);
        }

        if best_fitness > 0.99 {
            println!("🔥 ¡Evolución Exitosa! El organismo ha aprendido el patrón.");
            break;
        }
    }

    println!("\n✅ Evolución completada en {:?}", start_time.elapsed());
    let final_logits = organism.forward(&input);
    println!("Logits finales: {:?}", final_logits);
    println!("Probabilidades: {:?}", softmax(&final_logits));
}
