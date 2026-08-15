// =============================================================================
// tensor — Helpers de lectura de tensores desde la base redb
// =============================================================================
use crate::core::db::TENSOR_TABLE;
use crate::io::db_loader::NativeLoader;
use crate::nn::linear::GenomicLinear;
use redb::{ReadTransaction, ReadableTable};

impl NativeLoader {
    pub(crate) fn get_tensor(txn: &ReadTransaction, key: &str) -> Vec<u8> {
        if let Ok(t) = txn.open_table(TENSOR_TABLE) {
            if let Ok(Some(v)) = t.get(key) {
                return lz4_flex::decompress_size_prepended(v.value())
                    .unwrap_or_else(|_| v.value().to_vec());
            }
        }
        Vec::new()
    }

    pub(crate) fn get_tensor_f32(txn: &ReadTransaction, key: &str) -> Vec<f32> {
        let b = Self::get_tensor(txn, key);
        if b.is_empty() {
            return Vec::new();
        }
        let mut r = vec![0.0f32; b.len() / 4];
        unsafe {
            std::ptr::copy_nonoverlapping(b.as_ptr(), r.as_mut_ptr() as *mut u8, b.len());
        }
        r
    }

    pub(crate) fn get_linear(
        &self,
        txn: &ReadTransaction,
        p: &str,
        i_f: usize,
        o_f: usize,
        b_s: usize,
    ) -> GenomicLinear {
        let dna = Self::get_tensor(txn, &format!("{}.dna", p));
        let centroids = Self::get_tensor_f32(txn, &format!("{}.centroids", p));
        let anchors = Self::get_tensor(txn, &format!("{}.anchors", p));
        let bias = Self::get_tensor_f32(txn, &format!("{}.bias", p));
        let mask = Self::get_tensor(txn, &format!("{}.precision_mask", p));

        // Inferencia robusta de profundidad de bits basada en el tamaño real del buffer DNA
        let n_elements = i_f * o_f;
        let expected_2bit = (n_elements + 3) / 4;
        let expected_4bit = (n_elements + 1) / 2;

        let bit_depth = if dna.len() == n_elements * 4 {
            32
        } else if dna.len() == expected_4bit {
            4
        } else if dna.len() == expected_2bit {
            2
        } else {
            panic!("[Loader Critical] Tamaño de buffer DNA ({}) para capa '{}' no coincide con 2-bit ({}) ni 4-bit ({})",
                    dna.len(), p, expected_2bit, expected_4bit);
        };

        GenomicLinear::new(
            dna,
            anchors,
            centroids,
            o_f,
            i_f,
            b_s,
            Vec::new(), // explicitly ensure no internal norm
            1e-6,
            mask,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            bias,
            bit_depth,
        )
    }
}
