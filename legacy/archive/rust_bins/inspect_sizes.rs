use redb::{Database, ReadableTable, TableDefinition};
use std::env;

pub const TENSOR_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("tensors");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Uso: {} <ruta/al/modelo.gaje>", args[0]);
        std::process::exit(1);
    }
    let model_path = &args[1];
    let db = Database::open(model_path)?;
    let txn = db.begin_read()?;
    let table = txn.open_table(TENSOR_TABLE)?;
    let mut total_bytes = 0;

    println!(
        "=== Tensor Keys and Compressed Sizes for {} ===",
        model_path
    );
    for result in table.iter()? {
        let (key, value) = result?;
        let len = value.value().len();
        total_bytes += len;
        println!("{}: {} bytes (compressed)", key.value(), len);
    }
    println!(
        "Total compressed tensor bytes: {} bytes ({:.2} MB)",
        total_bytes,
        total_bytes as f64 / 1024.0 / 1024.0
    );
    Ok(())
}
