use redb::{Database, TableDefinition, ReadableTable};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex, Weak};
use std::collections::HashMap;
use std::path::Path;

pub const TENSOR_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("tensors");
pub const METADATA_TABLE: TableDefinition<&str, &str> = TableDefinition::new("metadata");
pub const MUTATIONS_TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("mutations");

// Registro global para evitar "Database already open"
static DB_REGISTRY: Mutex<Option<HashMap<String, Weak<Database>>>> = Mutex::new(None);

pub(crate) fn get_or_create_db(path: &str, create: bool) -> Result<Arc<Database>, String> {
    let mut registry_lock = DB_REGISTRY.lock().unwrap();
    if registry_lock.is_none() {
        *registry_lock = Some(HashMap::new());
    }
    let registry = registry_lock.as_mut().unwrap();

    let abs_path = if let Ok(p) = Path::new(path).canonicalize() {
        p.to_string_lossy().to_string()
    } else {
        let p = Path::new(path);
        if p.is_absolute() {
            p.to_string_lossy().to_string()
        } else {
            std::env::current_dir()
                .map(|cwd| cwd.join(p).to_string_lossy().to_string())
                .unwrap_or_else(|_| path.to_string())
        }
    };

    if let Some(weak_db) = registry.get(&abs_path) {
        if let Some(db) = weak_db.upgrade() {
            return Ok(db);
        }
    }

    let db = if create {
        Database::create(path).map_err(|e| e.to_string())?
    } else {
        Database::open(path).map_err(|e| e.to_string())?
    };

    let arc_db = Arc::new(db);
    registry.insert(abs_path, Arc::downgrade(&arc_db));
    Ok(arc_db)
}

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
        let db_arc = get_or_create_db(path, true).map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;
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
        let db_arc = get_or_create_db(path, false).map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;
        Ok(Self { db: db_arc })
    }

    pub fn read_tensor(&self, key: &str) -> PyResult<Vec<u8>> {
        let read_txn = self.db.begin_read().map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let table = read_txn.open_table(TENSOR_TABLE).map_err(|e| pyo3::exceptions::PyKeyError::new_err(e.to_string()))?;
        if let Some(val) = table.get(key).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))? {
            let data = val.value();
            // Intentar descomprimir si parece LZ4 (primeros 4 bytes son el tamaño original)
            // lz4_flex::decompress_size_prepended retornará error si no es un formato LZ4 válido con prefijo de tamaño
            match lz4_flex::decompress_size_prepended(data) {
                Ok(decompressed) => Ok(decompressed),
                Err(_) => Ok(data.to_vec()),
            }
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

    pub fn has_metadata(&self, key: &str) -> PyResult<bool> {
        let read_txn = self.db.begin_read().map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        if let Ok(table) = read_txn.open_table(METADATA_TABLE) {
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
