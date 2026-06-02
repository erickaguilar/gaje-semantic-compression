use crate::nn::spiking::layer::GajeNeuromorphicLayer;
use crate::compute::lagrangian::LagrangianEngine;
use crate::core::db::{GajeDatabaseWriter, METADATA_TABLE, TENSOR_TABLE};
use redb::ReadTransaction;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Smg1Config {
    pub vocab_size: usize,
    pub layer_dims: Vec<(usize, usize)>, // (num_neurons, weights_per_neuron)
    pub thresholds: Vec<f32>,
    pub decays: Vec<f32>,
}

pub struct Smg1Model {
    pub layers: Vec<GajeNeuromorphicLayer>,
    pub word_to_id: HashMap<String, usize>,
    pub id_to_word: Vec<String>,
}

pub fn save_smg1_model(
    path: &str,
    model: &Smg1Model,
    config: &Smg1Config,
) -> std::io::Result<()> {
    let mut writer = GajeDatabaseWriter::new(path).map_err(std::io::Error::other)?;
    let mut batch = writer.begin_batch().map_err(std::io::Error::other)?;

    // 1. Guardar Configuración y Vocabulario
    batch.write_metadata("config", &serde_json::to_string(config).unwrap()).unwrap();
    batch.write_metadata("vocabulary", &serde_json::to_string(&model.id_to_word).unwrap()).unwrap();

    let compress = |d: &[u8]| lz4_flex::compress_prepend_size(d);
    let f32_u8 = |d: &[f32]| unsafe { std::slice::from_raw_parts(d.as_ptr() as *const u8, d.len() * 4) };

    // 2. Guardar Capas
    for (i, layer) in model.layers.iter().enumerate() {
        let p = format!("layer.{}", i);
        batch.write_tensor(&format!("{}.weights", p), &compress(&layer.packed_weights)).unwrap();
        batch.write_tensor(&format!("{}.potentials_real", p), &compress(f32_u8(&layer.membrane_potentials_real))).unwrap();
        batch.write_tensor(&format!("{}.potentials_imag", p), &compress(f32_u8(&layer.membrane_potentials_imag))).unwrap();
        batch.write_tensor(&format!("{}.thresholds", p), &compress(f32_u8(&layer.thresholds))).unwrap();
        batch.write_tensor(&format!("{}.decays", p), &compress(f32_u8(&layer.decays))).unwrap();
        batch.write_tensor(&format!("{}.anchors", p), &compress(&layer.anchors_sparse_buffer())).unwrap();
    }

    batch.commit().map_err(std::io::Error::other)?;
    writer.compact().map_err(std::io::Error::other)?;
    Ok(())
}

pub fn load_smg1_model(path: &str) -> std::io::Result<(Smg1Model, Smg1Config)> {
    let db = crate::core::db::get_or_create_db(path, true).map_err(std::io::Error::other)?;
    let read_txn = db.begin_read().map_err(|e| std::io::Error::other(e.to_string()))?;

    // 1. Cargar Configuración y Vocabulario
    let config_json = {
        let table = read_txn.open_table(METADATA_TABLE).map_err(|e| std::io::Error::other(e.to_string()))?;
        table.get("config").map_err(|e| std::io::Error::other(e.to_string()))?
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "config not found"))?
            .value().to_string()
    };
    let config: Smg1Config = serde_json::from_str(&config_json)?;

    let id_to_word: Vec<String> = {
        let table = read_txn.open_table(METADATA_TABLE).map_err(|e| std::io::Error::other(e.to_string()))?;
        let vocab_json = table.get("vocabulary").map_err(|e| std::io::Error::other(e.to_string()))?
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "vocabulary not found"))?
            .value().to_string();
        serde_json::from_str(&vocab_json)?
    };

    let mut word_to_id = HashMap::new();
    for (i, word) in id_to_word.iter().enumerate() {
        word_to_id.insert(word.clone(), i);
    }

    // 2. Cargar Capas
    let mut layers = Vec::new();
    for i in 0..config.layer_dims.len() {
        let (num_neurons, weights_per_neuron) = config.layer_dims[i];
        let p = format!("layer.{}", i);
        
        let packed_weights = get_tensor(&read_txn, &format!("{}.weights", p));
        let mut membrane_potentials_real = get_tensor_f32(&read_txn, &format!("{}.potentials_real", p));
        let mut membrane_potentials_imag = get_tensor_f32(&read_txn, &format!("{}.potentials_imag", p));
        
        // Fallback para modelos antiguos
        if membrane_potentials_real.is_empty() {
            membrane_potentials_real = get_tensor_f32(&read_txn, &format!("{}.potentials", p));
        }
        if membrane_potentials_imag.is_empty() {
            membrane_potentials_imag = vec![0.0; num_neurons];
        }

        let thresholds = get_tensor_f32(&read_txn, &format!("{}.thresholds", p));
        let decays = get_tensor_f32(&read_txn, &format!("{}.decays", p));
        let anchors_u8 = get_tensor(&read_txn, &format!("{}.anchors", p));

        let mut layer = GajeNeuromorphicLayer {
            membrane_potentials_real,
            membrane_potentials_imag,
            thresholds,
            decays,
            packed_weights,
            anchor_indices: Vec::new(),
            anchor_values: Vec::new(),
            anchor_row_ptrs: vec![0; num_neurons + 1],
            num_neurons,
            weights_per_neuron,
            k_wta: (num_neurons / 10).max(1),
            rms_ema: 1.0,
            lagrangian: LagrangianEngine::new(1.0),
        };
        layer.load_anchors_from_u8(&anchors_u8);
        layers.push(layer);
    }

    Ok((Smg1Model { layers, word_to_id, id_to_word }, config))
}

fn get_tensor(txn: &ReadTransaction, key: &str) -> Vec<u8> {
    if let Ok(t) = txn.open_table(TENSOR_TABLE) {
        if let Ok(Some(v)) = t.get(key) {
            return lz4_flex::decompress_size_prepended(v.value()).unwrap_or_else(|_| v.value().to_vec());
        }
    }
    Vec::new()
}

fn get_tensor_f32(txn: &ReadTransaction, key: &str) -> Vec<f32> {
    let b = get_tensor(txn, key);
    if b.is_empty() { return Vec::new(); }
    let mut r = vec![0.0f32; b.len() / 4];
    unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), r.as_mut_ptr() as *mut u8, b.len()); }
    r
}
