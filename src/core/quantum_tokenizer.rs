//! # 🧬 QuantumGenomicTokenizer (Native Rust Prototype)
//!
//! Implementa la matemática de superposición cuántica de 2-qubits y su isomorfismo
//! con las bases nucleótidas {A, C, G, T} de 2-bits, utilizando matrices de densidad
//! hermíticas 4x4 y colapso por regla de Born.

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Complex32 {
    pub re: f32,
    pub im: f32,
}

impl Complex32 {
    pub fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    pub fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    pub fn norm_sq(&self) -> f32 {
        self.re * self.re + self.im * self.im
    }

    pub fn conj(&self) -> Self {
        Self { re: self.re, im: -self.im }
    }

    pub fn mul(&self, other: &Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DensityMatrix4x4 {
    pub m: [[Complex32; 4]; 4],
}

impl DensityMatrix4x4 {
    pub fn from_state_vector(psi: &[Complex32; 4]) -> Self {
        let mut m = [[Complex32::zero(); 4]; 4];
        let mut norm_sq = 0.0;
        for c in psi.iter() {
            norm_sq += c.norm_sq();
        }
        let inv_norm = if norm_sq > 1e-9 { 1.0 / norm_sq.sqrt() } else { 1.0 };
        let normalized = [
            Complex32::new(psi[0].re * inv_norm, psi[0].im * inv_norm),
            Complex32::new(psi[1].re * inv_norm, psi[1].im * inv_norm),
            Complex32::new(psi[2].re * inv_norm, psi[2].im * inv_norm),
            Complex32::new(psi[3].re * inv_norm, psi[3].im * inv_norm),
        ];

        for i in 0..4 {
            for j in 0..4 {
                m[i][j] = normalized[i].mul(&normalized[j].conj());
            }
        }
        Self { m }
    }

    pub fn trace(&self) -> f32 {
        self.m[0][0].re + self.m[1][1].re + self.m[2][2].re + self.m[3][3].re
    }

    pub fn purity(&self) -> f32 {
        // Tr(ρ²)
        let mut tr_rho2 = 0.0;
        for i in 0..4 {
            for k in 0..4 {
                let p = self.m[i][k].mul(&self.m[k][i]);
                tr_rho2 += p.re;
            }
        }
        tr_rho2
    }

    pub fn collapse_with_context(&self, context: &[f32; 4]) -> (char, f32) {
        let bases = ['A', 'C', 'G', 'T'];
        let mut max_idx = 0;
        let mut max_prob = -1.0;

        let mut c_norm_sq = 0.0;
        for val in context.iter() {
            c_norm_sq += val * val;
        }
        let inv_c_norm = if c_norm_sq > 1e-9 { 1.0 / c_norm_sq.sqrt() } else { 1.0 };

        for i in 0..4 {
            let base_prob = self.m[i][i].re;
            let c_factor = (context[i] * inv_c_norm).powi(2);
            let combined = base_prob * c_factor;
            if combined > max_prob {
                max_prob = combined;
                max_idx = i;
            }
        }

        (bases[max_idx], max_prob)
    }
}

pub struct QuantumGenomicTokenizer {
    pub smoothing: f32,
}

impl QuantumGenomicTokenizer {
    pub fn new() -> Self {
        Self { smoothing: 0.05 }
    }

    pub fn encode_char(&self, ch: char) -> DensityMatrix4x4 {
        let code = ch as u32;
        let t1 = (code % 360) as f32 * std::f32::consts::PI / 180.0;
        let t2 = ((code * 7) % 360) as f32 * std::f32::consts::PI / 180.0;
        let t3 = ((code * 13) % 360) as f32 * std::f32::consts::PI / 180.0;
        let t4 = ((code * 23) % 360) as f32 * std::f32::consts::PI / 180.0;

        let psi = [
            Complex32::new(t1.cos() * (t1.cos().powi(2) + self.smoothing), t1.sin() * (t1.cos().powi(2) + self.smoothing)),
            Complex32::new(t2.cos() * (t2.sin().powi(2) + self.smoothing), t2.sin() * (t2.sin().powi(2) + self.smoothing)),
            Complex32::new(t3.cos() * (t3.cos().powi(2) + self.smoothing), t3.sin() * (t3.cos().powi(2) + self.smoothing)),
            Complex32::new(t4.cos() * (t4.sin().powi(2) + self.smoothing), t4.sin() * (t4.sin().powi(2) + self.smoothing)),
        ];

        DensityMatrix4x4::from_state_vector(&psi)
    }

    pub fn encode_text_to_dna(&self, text: &str, context: Option<&[f32; 4]>) -> String {
        let default_ctx = [0.5, 0.5, 0.5, 0.5];
        let ctx = context.unwrap_or(&default_ctx);

        let mut dna = String::with_capacity(text.len());
        for ch in text.chars() {
            let rho = self.encode_char(ch);
            let (base, _) = rho.collapse_with_context(ctx);
            dna.push(base);
        }
        dna
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_density_matrix_properties() {
        let tokenizer = QuantumGenomicTokenizer::new();
        let rho = tokenizer.encode_char('G');

        // 1. Traza debe ser unitaria
        let tr = rho.trace();
        assert!((tr - 1.0).abs() < 1e-4, "Traza(ρ) debe ser 1.0, obtenido: {}", tr);

        // 2. Pureza de estado puro debe ser ~1.0
        let pur = rho.purity();
        assert!((pur - 1.0).abs() < 1e-3, "Pureza de estado puro debe ser ~1.0, obtenido: {}", pur);
    }

    #[test]
    fn test_text_to_dna_encoding() {
        let tokenizer = QuantumGenomicTokenizer::new();
        let dna = tokenizer.encode_text_to_dna("GAJE", None);
        assert_eq!(dna.len(), 4);
        for c in dna.chars() {
            assert!(c == 'A' || c == 'C' || c == 'G' || c == 'T');
        }
    }
}
