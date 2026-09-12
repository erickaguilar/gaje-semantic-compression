use _impl::core::tokenizer::GajeTokenizer;
use _impl::io::gmem::cosine_similarity;
use _impl::nn::llm::GenomicLLM;
use _impl::nn::repl::load_model_and_tokenizer;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

struct Pair {
    id: &'static str,
    stored: &'static str,
    query: &'static str,
    is_positive: bool,
    #[allow(dead_code)]
    description: &'static str,
}

fn get_calibration_corpus() -> Vec<Pair> {
    vec![
        // Positivos (P1..P20)
        Pair { id: "P1", stored: "La capital de Francia es París.", query: "¿Cuál es la capital de Francia?", is_positive: true, description: "Factual directo" },
        Pair { id: "P2", stored: "El agua hierve a 100°C a nivel del mar.", query: "¿A qué temperatura hierve el agua?", is_positive: true, description: "Factual directo" },
        Pair { id: "P3", stored: "GAJE comprime modelos con cuantización Q4_0.", query: "¿Cómo funciona la compresión de GAJE?", is_positive: true, description: "Técnico" },
        Pair { id: "P4", stored: "Marte es el cuarto planeta del sistema solar.", query: "¿Qué posición ocupa Marte?", is_positive: true, description: "Factual directo" },
        Pair { id: "P5", stored: "El ADN contiene información genética.", query: "¿Qué almacena el ADN?", is_positive: true, description: "Factual directo" },
        Pair { id: "P6", stored: "Rust es un lenguaje de programación de sistemas.", query: "¿Para qué sirve Rust?", is_positive: true, description: "Técnico" },
        Pair { id: "P7", stored: "La fotosíntesis convierte luz en energía química.", query: "¿Cómo obtienen energía las plantas?", is_positive: true, description: "Científico" },
        Pair { id: "P8", stored: "Einstein formuló la teoría de la relatividad.", query: "¿Quién propuso la relatividad?", is_positive: true, description: "Factual" },
        Pair { id: "P9", stored: "El Everest es la montaña más alta del mundo.", query: "¿Cuál es el pico más alto?", is_positive: true, description: "Factual" },
        Pair { id: "P10", stored: "La Revolución Francesa comenzó en 1789.", query: "¿Cuándo empezó la Revolución Francesa?", is_positive: true, description: "Histórico" },
        Pair { id: "P11", stored: "Madrid es la capital de España.", query: "¿Dónde está la sede del gobierno español?", is_positive: true, description: "Factual indirecto" },
        Pair { id: "P12", stored: "Python es un lenguaje interpretado de alto nivel.", query: "¿Qué tipo de lenguaje es Python?", is_positive: true, description: "Técnico" },
        Pair { id: "P13", stored: "El corazón bombea sangre por el cuerpo.", query: "¿Cuál es la función del corazón?", is_positive: true, description: "Biológico" },
        Pair { id: "P14", stored: "Tokyo es la capital de Japón.", query: "¿Cuál es la ciudad principal de Japón?", is_positive: true, description: "Factual indirecto" },
        Pair { id: "P15", stored: "Los electrones tienen carga negativa.", query: "¿Qué carga tienen los electrones?", is_positive: true, description: "Físico" },
        Pair { id: "P16", stored: "Shakespeare escribió Hamlet.", query: "¿Quién es el autor de Hamlet?", is_positive: true, description: "Literario" },
        Pair { id: "P17", stored: "El Sol es una estrella de tipo G.", query: "¿Qué tipo de estrella es el Sol?", is_positive: true, description: "Astronómico" },
        Pair { id: "P18", stored: "La Segunda Guerra Mundial terminó en 1945.", query: "¿Cuándo acabó la Segunda Guerra Mundial?", is_positive: true, description: "Histórico" },
        Pair { id: "P19", stored: "El café contiene cafeína.", query: "¿Qué estimulante tiene el café?", is_positive: true, description: "Factual" },
        Pair { id: "P20", stored: "Los mamíferos son animales de sangre caliente.", query: "¿Cómo regulan la temperatura los mamíferos?", is_positive: true, description: "Biológico" },

        // Negativos / Trampas (N1..N20)
        Pair { id: "N1", stored: "La capital de Francia es París.", query: "¿Cuál es el mejor restaurante de Roma?", is_positive: false, description: "Mismo dominio geo, distinto tema" },
        Pair { id: "N2", stored: "El agua hierve a 100°C.", query: "¿Cuánta agua hay en el océano?", is_positive: false, description: "Misma palabra clave (agua)" },
        Pair { id: "N3", stored: "GAJE comprime modelos.", query: "¿Cómo se escribe una función en Python?", is_positive: false, description: "Técnico sin relación" },
        Pair { id: "N4", stored: "Marte es el cuarto planeta.", query: "¿Cuántos anillos tiene Saturno?", is_positive: false, description: "Planetas, distinto objeto" },
        Pair { id: "N5", stored: "El ADN contiene información genética.", query: "¿Qué comen los murciélagos?", is_positive: false, description: "Fuera de dominio" },
        Pair { id: "N6", stored: "Rust es un lenguaje de sistemas.", query: "¿Cuál es la capital de Australia?", is_positive: false, description: "Fuera de dominio" },
        Pair { id: "N7", stored: "La fotosíntesis produce oxígeno.", query: "¿Cuántas patas tiene una araña?", is_positive: false, description: "Fuera de dominio" },
        Pair { id: "N8", stored: "Einstein formuló la relatividad.", query: "¿Qué es la teoría de cuerdas?", is_positive: false, description: "Física, distinto concepto" },
        Pair { id: "N9", stored: "El Everest es la montaña más alta.", query: "¿Cuál es el río más largo?", is_positive: false, description: "Geografía, distinto objeto" },
        Pair { id: "N10", stored: "La Revolución Francesa comenzó en 1789.", query: "¿Cuándo fue la independencia de México?", is_positive: false, description: "Historia, distinto evento" },
        Pair { id: "N11", stored: "Madrid es la capital de España.", query: "¿Cuál es la moneda de Japón?", is_positive: false, description: "Fuera de dominio" },
        Pair { id: "N12", stored: "Python es interpretado.", query: "¿Qué es la programación funcional?", is_positive: false, description: "Técnico, distinto concepto" },
        Pair { id: "N13", stored: "El corazón bombea sangre.", query: "¿Cómo funcionan los pulmones?", is_positive: false, description: "Biológico, distinto órgano" },
        Pair { id: "N14", stored: "Tokyo es la capital de Japón.", query: "¿Cuál es la comida típica de Italia?", is_positive: false, description: "Fuera de dominio" },
        Pair { id: "N15", stored: "Los electrones tienen carga negativa.", query: "¿Cómo funciona la gravedad?", is_positive: false, description: "Física, distinta fuerza" },
        Pair { id: "N16", stored: "Shakespeare escribió Hamlet.", query: "¿Quién pintó la Mona Lisa?", is_positive: false, description: "Arte, distinto autor" },
        Pair { id: "N17", stored: "El Sol es una estrella de tipo G.", query: "¿Qué son los agujeros negros?", is_positive: false, description: "Astronomía, distinto objeto" },
        Pair { id: "N18", stored: "La Segunda Guerra Mundial terminó en 1945.", query: "¿Qué causó la Primera Guerra Mundial?", is_positive: false, description: "Historia, distinta guerra" },
        Pair { id: "N19", stored: "El café contiene cafeína.", query: "¿Qué vitaminas tiene la naranja?", is_positive: false, description: "Alimentario, distinto nutriente" },
        Pair { id: "N20", stored: "Los mamíferos son de sangre caliente.", query: "¿Cómo se reproducen los peces?", is_positive: false, description: "Biológico, distinta clase" },
    ]
}

fn compute_mu_vector(
    llm: &GenomicLLM,
    tokenizer: &GajeTokenizer,
    corpus_path: &str,
    max_sentences: usize,
) -> Vec<f32> {
    let dim = llm.dim();
    let mut sum_vec = vec![0.0f32; dim];
    let mut count = 0usize;

    if let Ok(file) = File::open(corpus_path) {
        let reader = BufReader::new(file);
        for line in reader.lines().flatten() {
            let text = if line.contains("\"text\":") {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&line) {
                    val.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string()
                } else {
                    line
                }
            } else {
                line
            };

            let clean = text.trim();
            if clean.len() < 10 {
                continue;
            }
            if let Ok(v) = llm.embed_text(clean, tokenizer) {
                for i in 0..dim {
                    sum_vec[i] += v[i];
                }
                count += 1;
                if count >= max_sentences {
                    break;
                }
            }
        }
    }

    if count > 0 {
        for val in sum_vec.iter_mut() {
            *val /= count as f32;
        }
    }
    sum_vec
}

fn embed_whitened(
    llm: &GenomicLLM,
    tokenizer: &GajeTokenizer,
    text: &str,
    mu: &[f32],
) -> Vec<f32> {
    let raw = llm.embed_text(text, tokenizer).unwrap_or_else(|_| vec![0.0; llm.dim()]);
    let dim = llm.dim();
    let mut centered = vec![0.0f32; dim];
    let mut norm_sq = 0.0f32;

    for i in 0..dim {
        let diff = raw[i] - mu[i];
        centered[i] = diff;
        norm_sq += diff * diff;
    }
    let norm = norm_sq.sqrt().max(1e-8);
    for val in centered.iter_mut() {
        *val /= norm;
    }
    centered
}

struct Stats {
    mean: f32,
    std: f32,
    p10: f32,
    p90: f32,
}

fn compute_stats(mut values: Vec<f32>) -> Stats {
    if values.is_empty() {
        return Stats { mean: 0.0, std: 0.0, p10: 0.0, p90: 0.0 };
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();
    let sum: f32 = values.iter().sum();
    let mean = sum / n as f32;
    let var: f32 = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
    let std = var.sqrt();

    let idx10 = ((n as f32 * 0.10).floor() as usize).min(n - 1);
    let idx90 = ((n as f32 * 0.90).floor() as usize).min(n - 1);

    Stats {
        mean,
        std,
        p10: values[idx10],
        p90: values[idx90],
    }
}


struct Evaluation {
    tau_star: f32,
    margin_means: f32,
    margin_percentiles: f32,
    stat_pos: Stats,
    stat_neg: Stats,
    pos_items: Vec<(&'static str, f32)>,
    neg_items: Vec<(&'static str, f32)>,
}

fn evaluate_model_mode(
    llm: &GenomicLLM,
    tokenizer: &GajeTokenizer,
    corpus: &[Pair],
    mu_opt: Option<&[f32]>,
) -> Evaluation {
    let mut pos_sims = Vec::new();
    let mut neg_sims = Vec::new();
    let mut pos_items = Vec::new();
    let mut neg_items = Vec::new();

    for pair in corpus {
        let v_a = if let Some(mu) = mu_opt {
            embed_whitened(llm, tokenizer, pair.stored, mu)
        } else {
            llm.embed_text(pair.stored, tokenizer).unwrap()
        };

        let v_b = if let Some(mu) = mu_opt {
            embed_whitened(llm, tokenizer, pair.query, mu)
        } else {
            llm.embed_text(pair.query, tokenizer).unwrap()
        };

        let sim = cosine_similarity(&v_a, &v_b);
        if pair.is_positive {
            pos_sims.push(sim);
            pos_items.push((pair.id, sim));
        } else {
            neg_sims.push(sim);
            neg_items.push((pair.id, sim));
        }
    }

    let stat_pos = compute_stats(pos_sims);
    let stat_neg = compute_stats(neg_sims);

    let tau_star = (stat_pos.p10 + stat_neg.p90) / 2.0;
    let margin_means = stat_pos.mean - stat_neg.mean;
    let margin_percentiles = stat_pos.p10 - stat_neg.p90;

    Evaluation {
        tau_star,
        margin_means,
        margin_percentiles,
        stat_pos,
        stat_neg,
        pos_items,
        neg_items,
    }
}

#[test]
fn test_calibrate_all_models_whitening_comparison() {
    let models = [
        ("models/born/max.gaje", "max.gaje (Llama Q2_0 256d)"),
        ("models/production/qwen2_5_0_5b_q2_0.gaje", "qwen2_5_0_5b_q2_0.gaje (Qwen2.5 Q2_0 896d)"),
        ("models/production/gaje_pico_135m.gaje", "gaje_pico_135m.gaje (SmolLM FP32 576d)"),
    ];

    let corpus = get_calibration_corpus();
    assert_eq!(corpus.len(), 40);

    println!("\n=========================================================================================");
    println!("🔬 PROTOCOLO DE CALIBRACIÓN DE UMBRALES POR MODELO (40 PARES)");
    println!("=========================================================================================\n");

    for (path, label) in &models {
        if !Path::new(path).exists() {
            println!("⚠️ Saltando {} (no existe)", path);
            continue;
        }

        println!("-----------------------------------------------------------------------------------------");
        println!("📦 Evaluando: {}", label);
        println!("-----------------------------------------------------------------------------------------");

        let (llm, tokenizer) = load_model_and_tokenizer(path).expect("cargar modelo y tokenizer");

        // 1. Evaluación SIN whitening (Standard Weighted Pooling)
        let eval_raw = evaluate_model_mode(&llm, &tokenizer, &corpus, None);

        // 2. Cálculo de mu vector sobre corpus representativo (latam_corpus.jsonl)
        let mu = compute_mu_vector(&llm, &tokenizer, "data/latam_corpus.jsonl", 150);
        let eval_white = evaluate_model_mode(&llm, &tokenizer, &corpus, Some(&mu));

        let diag_raw = if eval_raw.margin_percentiles > 0.15 {
            "🟢 Separación limpia"
        } else if eval_raw.margin_percentiles > 0.08 {
            "🟡 Separación aceptable"
        } else if eval_raw.margin_percentiles > 0.0 {
            "🟠 Separación estrecha"
        } else {
            "🔴 Solapamiento"
        };

        let diag_white = if eval_white.margin_percentiles > 0.15 {
            "🟢 Separación limpia"
        } else if eval_white.margin_percentiles > 0.08 {
            "🟡 Separación aceptable"
        } else if eval_white.margin_percentiles > 0.0 {
            "🟠 Separación estrecha"
        } else {
            "🔴 Solapamiento"
        };

        println!("  [1] Sin Whitening (Weighted Pooling):");
        println!("      • Positivos:  μ={:.4}, σ={:.4}, p10={:.4}", eval_raw.stat_pos.mean, eval_raw.stat_pos.std, eval_raw.stat_pos.p10);
        println!("      • Negativos:  μ={:.4}, σ={:.4}, p90={:.4}", eval_raw.stat_neg.mean, eval_raw.stat_neg.std, eval_raw.stat_neg.p90);
        println!("      • Δ Medias:   {:.4}", eval_raw.margin_means);
        println!("      • Margen p:   {:.4} ({})", eval_raw.margin_percentiles, diag_raw);
        println!("      • τ* óptimo:  {:.4}", eval_raw.tau_star);

        println!("  [2] Con Whitening (μ Centering + L2 Normalization):");
        println!("      • Positivos:  μ={:.4}, σ={:.4}, p10={:.4}", eval_white.stat_pos.mean, eval_white.stat_pos.std, eval_white.stat_pos.p10);
        println!("      • Negativos:  μ={:.4}, σ={:.4}, p90={:.4}", eval_white.stat_neg.mean, eval_white.stat_neg.std, eval_white.stat_neg.p90);
        println!("      • Δ Medias:   {:.4}", eval_white.margin_means);
        println!("      • Margen p:   {:.4} ({})", eval_white.margin_percentiles, diag_white);
        println!("      • τ* óptimo:  {:.4}", eval_white.tau_star);

        let delta_margin = eval_white.margin_percentiles - eval_raw.margin_percentiles;
        println!("  ⚡ Ganancia de Margen por Whitening: {:+.4}", delta_margin);

        // Imprimir pares atípicos
        let mut sorted_raw_neg = eval_raw.neg_items.clone();
        sorted_raw_neg.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        println!("      ⚠️ Top 3 Negativos más altos (Raw):");
        for (id, sim) in sorted_raw_neg.iter().take(3) {
            println!("         • {}: cos={:.4}", id, sim);
        }

        let mut sorted_raw_pos = eval_raw.pos_items.clone();
        sorted_raw_pos.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        println!("      ⚠️ Bottom 3 Positivos más bajos (Raw):");
        for (id, sim) in sorted_raw_pos.iter().take(3) {
            println!("         • {}: cos={:.4}", id, sim);
        }
        println!();
    }
}
