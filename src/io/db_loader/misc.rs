// =============================================================================
// misc — load_tokenizer y list_mutations
// =============================================================================
use crate::core::tokenizer::GajeTokenizer;
use crate::io::db_loader::NativeLoader;
use redb::ReadableTable;

impl NativeLoader {
    pub fn load_tokenizer(&self) -> std::io::Result<GajeTokenizer> {
        let reader = crate::core::db::GajeDatabaseReader {
            db: self.db.clone(),
        };
        let json_str = reader
            .read_metadata_core("tokenizer")
            .map_err(std::io::Error::other)?;
        GajeTokenizer::from_bytes(json_str.as_bytes())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))
    }

    pub fn list_mutations(&self) -> std::io::Result<Vec<(u64, Vec<u8>)>> {
        let read_txn = self
            .db
            .begin_read()
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        if let Ok(table) = read_txn.open_table(crate::core::db::MUTATIONS_TABLE) {
            let mut res = Vec::new();
            for (ts, val) in table
                .iter()
                .map_err(|e| std::io::Error::other(e.to_string()))?
                .flatten()
            {
                res.push((ts.value(), val.value().to_vec()));
            }
            Ok(res)
        } else {
            Ok(Vec::new())
        }
    }
}
