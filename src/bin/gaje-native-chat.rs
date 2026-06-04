use _impl::core::sdk::GajeSession;
use std::io::{self, Write};

/// # 🧬 GAJE Native Chat: Demo de Soberanía Total
///
/// Este binario demuestra el uso del SDK nativo sin ninguna dependencia de Python.
/// Ejecuta un bucle de chat interactivo utilizando la lógica de sesión persistente
/// y memoria toroidal integrada en Rust.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        println!("❌ Error: Se requiere el path al modelo.");
        println!("Uso: gaje-native-chat <path_al_modelo.gaje>");
        return Ok(());
    }

    println!("--- 🏛️ GAJE-Core Native SDK Demo ---");
    println!("[*] Inicializando motor genómico (Zero-GIL)...");

    // Cargar sesión con capacidad de 20 interacciones en memoria toroidal
    let mut session = GajeSession::load(&args[1], 20)?;

    println!("✅ Soberanía Nativa alcanzada. Listo para conversar.");
    println!("[!] Escribe 'exit' para salir.");

    loop {
        print!("\n👤 Usuario > ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() || input == "exit" || input == "quit" {
            break;
        }

        let start = std::time::Instant::now();

        // El método 'chat' maneja internamente:
        // 1. Tokenización
        // 2. Recuperación de memoria semántica
        // 3. Inferencia LLM genómica
        // 4. Muestreo toroidal
        // 5. Guardado en memoria toroidal
        match session.chat(input, 128, 0.7, 0.9) {
            Ok(response) => {
                let duration = start.elapsed();
                println!("🧬 Organismo > {}", response);
                let sparsity = _impl::compute::diagnostics::get_sparsity_report();
                println!("\n   [Latencia Nativa: {:.2?}] | [Sparsity Temporal: {:.2}%]", duration, sparsity);
            }
            Err(e) => {
                println!("❌ Error en el motor: {}", e);
            }
        }
    }

    println!("\n[*] Organismo hibernando. SDK cerrado exitosamente.");
    Ok(())
}
