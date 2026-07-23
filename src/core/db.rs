#[cfg(feature = "python")]
use pyo3::prelude::*;
use redb::{Database, TableDefinition};
use std::sync::Arc;

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
pub struct GajeDatabaseReader {
    pub db: Arc<Database>,
}

#[cfg_attr(feature = "python", pymethods)]
impl GajeDatabaseReader {
    #[cfg(feature = "python")]
    #[new]
    pub fn py_new(path: &str) -> PyResult<Self> {
        let db = get_or_create_db(path, true).map_err(pyo3::exceptions::PyIOError::new_err)?;
        Ok(GajeDatabaseReader { db })
    }

    #[cfg(feature = "python")]
    pub fn read_metadata(&self, key: &str) -> PyResult<String> {
        self.read_metadata_core(key)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    #[cfg(feature = "python")]
    pub fn has_metadata(&self, key: &str) -> PyResult<bool> {
        self.has_metadata_core(key)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    #[cfg(feature = "python")]
    pub fn read_tensor(&self, key: &str) -> PyResult<Vec<u8>> {
        let read_txn = self
            .db
            .begin_read()
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let table = read_txn
            .open_table(TENSOR_TABLE)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let val = table
            .get(key)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("Tensor key {} not found", key))
            })?;

        let data = val.value();
        match lz4_flex::decompress_size_prepended(data) {
            Ok(decompressed) => Ok(decompressed),
            Err(_) => Ok(data.to_vec()),
        }
    }

    #[cfg(feature = "python")]
    pub fn has_tensor(&self, key: &str) -> PyResult<bool> {
        let read_txn = self
            .db
            .begin_read()
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let table = read_txn
            .open_table(TENSOR_TABLE)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let val = table
            .get(key)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(val.is_some())
    }
}

impl GajeDatabaseReader {
    pub fn read_metadata_core(&self, key: &str) -> Result<String, String> {
        let read_txn = self.db.begin_read().map_err(|e| e.to_string())?;
        let table = match read_txn.open_table(METADATA_TABLE) {
            Ok(t) => t,
            Err(e) => return Err(format!("Error opening metadata table: {}", e)),
        };
        let val = table
            .get(key)
            .map_err(|e| e.to_string())?
            .ok_or_else(|| format!("Metadata key {} not found", key))?;
        Ok(val.value().to_string())
    }

    pub fn has_metadata_core(&self, key: &str) -> Result<bool, String> {
        let read_txn = self.db.begin_read().map_err(|e| e.to_string())?;
        let table = match read_txn.open_table(METADATA_TABLE) {
            Ok(t) => t,
            Err(_) => return Ok(false),
        };
        let val = table.get(key).map_err(|e| e.to_string())?;
        Ok(val.is_some())
    }
}

#[cfg_attr(feature = "python", pyclass)]
pub struct GajeDatabaseWriter {
    pub db: Arc<Database>,
}

#[cfg_attr(feature = "python", pymethods)]
impl GajeDatabaseWriter {
    #[cfg(feature = "python")]
    #[new]
    pub fn py_new(path: &str) -> PyResult<Self> {
        let db = get_or_create_db(path, false).map_err(pyo3::exceptions::PyIOError::new_err)?;
        Ok(GajeDatabaseWriter { db })
    }

    #[cfg(feature = "python")]
    #[pyo3(name = "write_metadata")]
    pub fn py_write_metadata(&self, key: &str, value: &str) -> PyResult<()> {
        let mut writer = self.begin_batch_rust().map_err(pyo3::exceptions::PyIOError::new_err)?;
        writer.write_metadata(key, value).map_err(pyo3::exceptions::PyIOError::new_err)?;
        writer.commit_core().map_err(pyo3::exceptions::PyIOError::new_err)
    }

    #[cfg(feature = "python")]
    #[pyo3(name = "write_tensor_compressed")]
    pub fn py_write_tensor_compressed(&self, key: &str, data: &[u8]) -> PyResult<()> {
        let mut writer = self.begin_batch_rust().map_err(pyo3::exceptions::PyIOError::new_err)?;
        writer.write_tensor_compressed(key, data).map_err(pyo3::exceptions::PyIOError::new_err)?;
        writer.commit_core().map_err(pyo3::exceptions::PyIOError::new_err)
    }

    #[cfg(feature = "python")]
    pub fn begin_batch(&self) -> PyResult<GajeBatchWriter> {
        self.begin_batch_rust().map_err(pyo3::exceptions::PyIOError::new_err)
    }
}

#[cfg_attr(feature = "python", pyclass)]
pub struct GajeBatchWriter {
    pub txn: Option<redb::WriteTransaction>,
}

#[cfg_attr(feature = "python", pymethods)]
impl GajeBatchWriter {
    #[cfg(feature = "python")]
    #[pyo3(name = "write_tensor")]
    pub fn py_write_tensor(&mut self, key: &str, data: &[u8]) -> PyResult<()> {
        self.write_tensor(key, data).map_err(pyo3::exceptions::PyIOError::new_err)
    }

    #[cfg(feature = "python")]
    #[pyo3(name = "write_tensor_compressed")]
    pub fn py_write_tensor_compressed(&mut self, key: &str, data: &[u8]) -> PyResult<()> {
        self.write_tensor_compressed(key, data).map_err(pyo3::exceptions::PyIOError::new_err)
    }

    #[cfg(feature = "python")]
    #[pyo3(name = "write_metadata")]
    pub fn py_write_metadata(&mut self, key: &str, value: &str) -> PyResult<()> {
        self.write_metadata(key, value).map_err(pyo3::exceptions::PyIOError::new_err)
    }

    #[cfg(feature = "python")]
    #[pyo3(name = "commit")]
    pub fn py_commit(&mut self) -> PyResult<()> {
        self.commit_core().map_err(pyo3::exceptions::PyIOError::new_err)
    }
}

impl GajeBatchWriter {
    pub fn write_tensor(&mut self, key: &str, data: &[u8]) -> Result<(), String> {
        let txn = self.txn.as_mut().ok_or("Transaction closed")?;
        let mut table = txn
            .open_table(TENSOR_TABLE)
            .map_err(|e| e.to_string())?;
        table.insert(key, data).map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn write_tensor_compressed(&mut self, key: &str, data: &[u8]) -> Result<(), String> {
        let txn = self.txn.as_mut().ok_or("Transaction closed")?;
        let mut table = txn
            .open_table(TENSOR_TABLE)
            .map_err(|e| e.to_string())?;
        let compressed = lz4_flex::compress_prepend_size(data);
        table.insert(key, compressed.as_slice()).map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn write_metadata(&mut self, key: &str, value: &str) -> Result<(), String> {
        let txn = self.txn.as_mut().ok_or("Transaction closed")?;
        let mut table = txn
            .open_table(METADATA_TABLE)
            .map_err(|e| e.to_string())?;
        table.insert(key, value).map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn commit(mut self) -> Result<(), String> {
        self.commit_core()
    }

    pub fn commit_core(&mut self) -> Result<(), String> {
        let txn = self.txn.take().ok_or("Transaction already committed")?;
        txn.commit().map_err(|e| e.to_string())?;
        Ok(())
    }
}

impl GajeDatabaseWriter {
    pub fn new(path: &str) -> Result<Self, String> {
        let db = get_or_create_db(path, false)?;
        Ok(GajeDatabaseWriter { db })
    }
    pub fn begin_batch_rust(&self) -> Result<GajeBatchWriter, String> {
        let txn = self.db.begin_write().map_err(|e| e.to_string())?;
        Ok(GajeBatchWriter { txn: Some(txn) })
    }
    pub fn compact(&self) -> Result<(), String> {
        Ok(())
    }
}
