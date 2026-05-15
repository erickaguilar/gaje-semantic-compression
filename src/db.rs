use redb::{Database, TableDefinition};
use pyo3::prelude::*;

const TENSOR_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("tensors");
const METADATA_TABLE: TableDefinition<&str, &str> = TableDefinition::new("metadata");

#[pyclass]
pub struct GajeDatabaseWriter {
    db: Database,
}

#[pymethods]
impl GajeDatabaseWriter {
    #[new]
    pub fn new(path: &str) -> PyResult<Self> {
        let db = Database::create(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let write_txn = db.begin_write().map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        {
            write_txn.open_table(TENSOR_TABLE).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            write_txn.open_table(METADATA_TABLE).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }
        write_txn.commit().map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(Self { db })
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
    db: Database,
}

#[pymethods]
impl GajeDatabaseReader {
    #[new]
    pub fn new(path: &str) -> PyResult<Self> {
        let db = Database::open(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(Self { db })
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
}
