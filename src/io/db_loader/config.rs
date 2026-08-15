// =============================================================================
// config — Apertura de la base y carga de configuración del modelo
// =============================================================================
use crate::io::config::ModelConfig;
use crate::io::db_loader::NativeLoader;

impl NativeLoader {
    pub fn new(path: &str) -> std::io::Result<Self> {
        Self::new_with_mode(path, true)
    }

    pub fn new_with_mode(path: &str, read_only: bool) -> std::io::Result<Self> {
        Ok(NativeLoader {
            db: crate::core::db::get_or_create_db(path, read_only)
                .map_err(std::io::Error::other)?,
        })
    }

    pub fn load_config(&self) -> std::io::Result<ModelConfig> {
        let reader = crate::core::db::GajeDatabaseReader {
            db: self.db.clone(),
        };
        let json_str = reader
            .read_metadata_core("config")
            .map_err(std::io::Error::other)?;
        Ok(serde_json::from_str(&json_str)?)
    }
}
