use redb::{Database, TableDefinition};
use std::sync::Arc;
#[cfg(feature = "python")]
use pyo3::prelude::*;

pub const TENSOR_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("tensors");
pub const METADATA_TABLE: TableDefinition<&str, &str> = TableDefinition::new("metadata");
pub const MUTATIONS_TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("mutations");

pub fn get_or_create_db(path: &str, read_only: bool) -> Result<Arc<Database>, String> {
    let db = if read_only {
        Database::open(path).map_err(|e| e.to_string())?
    } else {
        Database::create(path).map_err(|e| e.to_string())?
    };
    Ok(Arc::new(db))
}

#[cfg_attr(feature = "python", pyclass)]
pub struct GajeDatabaseReader { pub db: Arc<Database> }

#[cfg_attr(feature = "python", pymethods)]
impl GajeDatabaseReader {
    #[cfg(feature = "python")]
    #[new]
    pub fn py_new(path: &str) -> PyResult<Self> {
        let db = get_or_create_db(path, true).map_err(pyo3::exceptions::PyIOError::new_err)?;
        Ok(GajeDatabaseReader { db })
    }

    #[cfg(feature = "python")]
    pub fn get_metadata(&self, key: &str) -> PyResult<Option<String>> {
        let read_txn = self.db.begin_read().map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let table = read_txn.open_table(METADATA_TABLE).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let val = table.get(key).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(val.map(|v| v.value().to_string()))
    }
}

#[cfg_attr(feature = "python", pyclass)]
pub struct GajeDatabaseWriter { pub db: Arc<Database> }

#[cfg_attr(feature = "python", pymethods)]
impl GajeDatabaseWriter {
    #[cfg(feature = "python")]
    #[new]
    pub fn py_new(path: &str) -> PyResult<Self> {
        let db = get_or_create_db(path, false).map_err(pyo3::exceptions::PyIOError::new_err)?;
        Ok(GajeDatabaseWriter { db })
    }

    #[cfg(feature = "python")]
    pub fn write_metadata(&self, key: &str, value: &str) -> PyResult<()> {
        let write_txn = self.db.begin_write().map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        {
            let mut table = write_txn.open_table(METADATA_TABLE).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            table.insert(key, value).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }
        write_txn.commit().map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(())
    }
}

pub struct GajeBatchWriter { pub txn: redb::WriteTransaction }

impl GajeBatchWriter {
    pub fn write_tensor(&mut self, key: &str, data: &[u8]) -> Result<(), String> {
        let mut table = self.txn.open_table(TENSOR_TABLE).map_err(|e| e.to_string())?;
        table.insert(key, data).map_err(|e| e.to_string())?;
        Ok(())
    }
    pub fn write_metadata(&mut self, key: &str, value: &str) -> Result<(), String> {
        let mut table = self.txn.open_table(METADATA_TABLE).map_err(|e| e.to_string())?;
        table.insert(key, value).map_err(|e| e.to_string())?;
        Ok(())
    }
    pub fn commit(self) -> Result<(), String> { self.txn.commit().map_err(|e| e.to_string())?; Ok(()) }
}

impl GajeDatabaseWriter {
    pub fn new(path: &str) -> Result<Self, String> {
        let db = get_or_create_db(path, false)?; Ok(GajeDatabaseWriter { db })
    }
    pub fn begin_batch(&mut self) -> Result<GajeBatchWriter, String> {
        let txn = self.db.begin_write().map_err(|e| e.to_string())?;
        Ok(GajeBatchWriter { txn })
    }
    pub fn compact(&self) -> Result<(), String> { Ok(()) }
}
