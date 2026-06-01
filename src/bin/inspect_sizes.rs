use redb::{Database, TableDefinition, ReadableTable};

pub const TENSOR_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("tensors");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db = Database::open("models/micro_distilled_coherence.gaje")?;
    let txn = db.begin_read()?;
    let table = txn.open_table(TENSOR_TABLE)?;
    let mut total_bytes = 0;
    
    println!("=== Tensor Keys and Compressed Sizes ===");
    for result in table.iter()? {
        let (key, value) = result?;
        let len = value.value().len();
        total_bytes += len;
        println!("{}: {} bytes (compressed)", key.value(), len);
    }
    println!("Total compressed tensor bytes: {} bytes ({:.2} MB)", total_bytes, total_bytes as f64 / 1024.0 / 1024.0);
    Ok(())
}
