use redb::{Database, TableDefinition, ReadableTable};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub const TENSOR_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("tensors");
pub const METADATA_TABLE: TableDefinition<&str, &str> = TableDefinition::new("metadata");
pub const MUTATIONS_TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("mutations");

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Mutation {
    pub layer_name: String,
    pub delta_centroids: Vec<f32>,
    pub delta_epi_centroids: Vec<f32>,
    pub delta_tri_centroids: Vec<f32>,
}

#[pyclass]
pub struct GajeDatabaseWriter {
    pub(crate) db: Arc<Database>,
}

#[pymethods]
impl GajeDatabaseWriter {
    #[new]
    pub fn new(path: &str) -> PyResult<Self> {
        let db = Database::create(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let db_arc = Arc::new(db);
        let write_txn = db_arc.begin_write().map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        {
            write_txn.open_table(TENSOR_TABLE).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            write_txn.open_table(METADATA_TABLE).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            write_txn.open_table(MUTATIONS_TABLE).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }
        write_txn.commit().map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(Self { db: db_arc })
    }

    pub fn compact(&self) -> PyResult<bool> {
        // Redb needs exclusive access for compaction.
        if let Some(db_mut) = Arc::get_mut(&mut Arc::clone(&self.db)) {
             db_mut.compact().map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err("Cannot compact database: multiple references exist"))
        }
    }

    pub fn write_mutation(&self, timestamp: u64, data: &[u8]) -> PyResult<()> {
        let write_txn = self.db.begin_write().map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        {
            let mut table = write_txn.open_table(MUTATIONS_TABLE).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            table.insert(timestamp, data).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }
        write_txn.commit().map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(())
    }

    pub fn write_tensor(&self, key: &str, data: &[u8]) -> PyResult<()> {
        let write_txn = self.db.begin_write().map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        {
            let mut table = write_txn.open_table(TENSOR_TABLE).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            table.insert(key, data).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }
        write_txn.commit().map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(())
    }

    pub fn write_metadata(&self, key: &str, json_str: &str) -> PyResult<()> {
        let write_txn = self.db.begin_write().map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        {
            let mut table = write_txn.open_table(METADATA_TABLE).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            table.insert(key, json_str).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }
        write_txn.commit().map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(())
    }
}

#[pyclass]
pub struct GajeDatabaseReader {
    pub(crate) db: Arc<Database>,
}

impl GajeDatabaseReader {
    pub fn new_from_db(db: Arc<Database>) -> Self {
        Self { db }
    }
}

#[pymethods]
impl GajeDatabaseReader {
    #[new]
    pub fn new(path: &str) -> PyResult<Self> {
        let db = Database::open(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(Self { db: Arc::new(db) })
    }

    pub fn read_tensor(&self, key: &str) -> PyResult<Vec<u8>> {
        let read_txn = self.db.begin_read().map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let table = read_txn.open_table(TENSOR_TABLE).map_err(|e| pyo3::exceptions::PyKeyError::new_err(e.to_string()))?;
        if let Some(val) = table.get(key).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))? {
            Ok(val.value().to_vec())
        } else {
            Err(pyo3::exceptions::PyKeyError::new_err(format!("Key not found: {}", key)))
        }
    }
    
    pub fn has_tensor(&self, key: &str) -> PyResult<bool> {
        let read_txn = self.db.begin_read().map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        if let Ok(table) = read_txn.open_table(TENSOR_TABLE) {
            if let Ok(Some(_)) = table.get(key) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub fn read_metadata(&self, key: &str) -> PyResult<String> {
        let read_txn = self.db.begin_read().map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let table = read_txn.open_table(METADATA_TABLE).map_err(|e| pyo3::exceptions::PyKeyError::new_err(e.to_string()))?;
        if let Some(val) = table.get(key).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))? {
            Ok(val.value().to_string())
        } else {
            Err(pyo3::exceptions::PyKeyError::new_err(format!("Key not found: {}", key)))
        }
    }

    pub fn list_mutations(&self) -> PyResult<Vec<(u64, Vec<u8>)>> {
        let read_txn = self.db.begin_read().map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let table = match read_txn.open_table(MUTATIONS_TABLE) {
            Ok(t) => t,
            Err(_) => return Ok(Vec::new()), // Table doesn't exist yet, so no mutations
        };
        let mut results = Vec::new();
        let iter = table.iter().map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        for res in iter {
            let (k, v) = res.map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            results.push((k.value(), v.value().to_vec()));
        }
        Ok(results)
    }
}
