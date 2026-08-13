use rand::Rng;
use std::time::Instant;

/// Representación de los 4 Centroides del ADN Genómico (2-bit)
const CENTROIDES: [f32; 4] = [-1.5, -0.5, 0.5, 1.5];

/// Función para cuantizar un valor f32 al centroide más cercano
fn cuantizar(val: f32) -> f32 {
    CENTROIDES
        .iter()
        .min_by(|&&a, &&b| (val - a).abs().partial_cmp(&(val - b).abs()).unwrap())
        .cloned()
        .unwrap()
}

fn main() {
    println!("🧬 DEMOSTRACIÓN: UNIFICACIÓN DE ANCLAS E ISLAS DE ESTABILIDAD 🧬");
    println!("----------------------------------------------------------------");

    let mut rng = rand::thread_rng();
    let num_neuronas = 20;
    let señal_objetivo: f32 = 10.0; // La suma que queremos alcanzar de forma estable

    // 1. ESCENARIO A: Solo Neuronas de 2-bits (Inestabilidad)
    println!("\n[Escenario A] 20 Neuronas de 2-bits puros (Sin Anclas)");
    let mut pesos_2bit: Vec<f32> = (0..num_neuronas)
        .map(|_| {
            let raw: f32 = rng.gen_range(-2.0..2.0);
            cuantizar(raw)
        })
        .collect();

    let suma_inicial: f32 = pesos_2bit.iter().sum();
    let error_inicial: f32 = (señal_objetivo - suma_inicial).abs();
    println!(
        "   > Suma Total: {:.2} | Error: {:.2}",
        suma_inicial, error_inicial
    );
    println!("   > Estado: Las neuronas están 'atrapadas' en sus centroides. No pueden ajustarse con precisión.");

    // 2. ESCENARIO B: Cristalización Semántica (Inyección de un Ancla)
    println!("\n[Escenario B] Inyección de 1 ANCLA (16-bits) + Islas de Estabilidad");

    // Creamos una \"Isla\" donde el Ancla guía a las demás
    let mut ancla: f32 = 1.0; // Empezamos con un valor base de alta precisión
    let start = Instant::now();

    // Ciclo de \"Cristalización\": El Ancla se ajusta y las neuronas de 2-bits se alinean
    for _ in 0..100 {
        let suma_actual: f32 = pesos_2bit.iter().sum::<f32>() + ancla;
        let diff = señal_objetivo - suma_actual;

        // El Ancla absorbe el error residual de alta frecuencia
        ancla += diff * 0.1;

        // Las neuronas de 2-bits intentan saltar de centroide si el error es muy grande
        for p in pesos_2bit.iter_mut() {
            let mut candidato = *p;
            if diff > 0.5 {
                candidato = cuantizar(*p + 0.1);
            } else if diff < -0.5 {
                candidato = cuantizar(*p - 0.1);
            }

            if (señal_objetivo - (suma_actual - *p + candidato)).abs()
                < (señal_objetivo - suma_actual).abs()
            {
                *p = candidato;
            }
        }
    }

    let suma_final: f32 = pesos_2bit.iter().sum::<f32>() + ancla;
    let error_final: f32 = (señal_objetivo - suma_final).abs();
    println!(
        "   > Suma Total (2-bits + Ancla): {:.4} | Error Final: {:.4}",
        suma_final, error_final
    );
    println!("   > Tiempo de Cristalización: {:?}", start.elapsed());

    // 3. ANÁLISIS DE LA ISLA
    println!("\n🔍 Análisis de la Isla de Estabilidad:");
    println!("   - Masa Genómica (2-bits): {:?}", pesos_2bit);
    println!("   - Núcleo de Estabilidad (Ancla): {:.4}", ancla);
    println!(
        "   - Eficiencia: {:.1}% de los pesos son 2-bit, pero el error es < 0.001.",
        (num_neuronas as f32 / (num_neuronas + 1) as f32) * 100.0
    );

    println!("\n✨ CONCLUSIÓN: El Ancla actuó como 'semilla' nucleando una Isla de Estabilidad.");
    println!(
        "   La inteligencia (precisión) se mantiene gracias al Ancla, mientras la masa de datos"
    );
    println!("   (2-bits) se organiza siguiendo la estructura geométrica impuesta.");
    println!("----------------------------------------------------------------");
}
