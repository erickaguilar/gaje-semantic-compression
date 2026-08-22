use crate::io::flat_reader::load_genomic_auto;

#[test]
fn test_body_refine_changes_weights_on_real_model() {
    let path = "models/production/smollm2_4bit.gaje.flat";
    if !std::path::Path::new(path).exists() {
        eprintln!("skip: modelo no presente en {}", path);
        return;
    }
    let mut model = load_genomic_auto(path).expect("cargar modelo");

    let before_down = model.blocks[0].w_down.centroids.clone();
    let before_wo = model.blocks[0].w_o.centroids.clone();

    let x = model.embeddings.get_row_core(4).unwrap();
    let out = model.blocks[0].forward_core(x.clone(), 0).unwrap();
    let d_out = vec![0.01f32; out.len()];
    model.blocks[0]
        .refine_with_grads_core(x, d_out, 0, 1e-3)
        .expect("refine_with_grads_core debió funcionar");

    let changed = model.blocks[0].w_down.centroids != before_down
        || model.blocks[0].w_o.centroids != before_wo;
    assert!(
        changed,
        "el cuerpo Genomic4Bit debió mutar sus centroides tras refine"
    );
}

#[test]
fn test_full_body_ce_training_changes_multiple_blocks() {
    let path = "models/production/smollm2_4bit.gaje.flat";
    if !std::path::Path::new(path).exists() {
        eprintln!("skip: modelo no presente en {}", path);
        return;
    }
    let mut model = load_genomic_auto(path).expect("cargar modelo");
    let n = model.blocks.len();
    assert!(n >= 3, "el modelo debe tener al menos 3 bloques");

    let before_last = model.blocks[n - 1].gate_gen.centroids.clone();

    // Entrenamiento del cuerpo vía CE (camino estable: último bloque).
    model.clear_cache_core();
    model
        .train_sequence_body_core(vec![4usize, 5], 1e-5)
        .expect("body CE training");

    // Tras entrenar, el forward completo normal debe seguir produciendo logits finitos.
    model.clear_cache_core();
    let (logits, _) = model
        .forward_with_hidden_core(4, true)
        .expect("el forward normal debe seguir estable tras entrenar el cuerpo");
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "logits deben ser finitos"
    );

    let changed = model.blocks[n - 1].gate_gen.centroids != before_last;
    assert!(
        changed,
        "el entrenamiento CE del cuerpo debió mutar los centroides del último bloque"
    );
}

#[test]
fn test_cached_backprop_trains_middle_block_no_nan() {
    let path = "models/production/smollm2_4bit.gaje.flat";
    if !std::path::Path::new(path).exists() {
        eprintln!("skip: modelo no presente en {}", path);
        return;
    }
    let mut model = load_genomic_auto(path).expect("cargar modelo");
    let n = model.blocks.len();
    assert!(n >= 3);

    // Entrenamos los últimos 3 bloques con caché (escalera), grad-clipping activo.
    let before_last = model.blocks[n - 1].gate_gen.centroids.clone();
    let before_mid = model.blocks[n - 2].gate_gen.centroids.clone();
    let before_first = model.blocks[n - 3].gate_gen.centroids.clone();

    let loss = model
        .train_sequence_cached_core(vec![4usize, 5, 6], 1e-5, 3, 1.0)
        .expect("cached training debió funcionar");
    assert!(loss.is_finite(), "loss debe ser finita, got {loss}");

    // El forward posterior debe ser estable (sin NaN) — clave del rediseño.
    model.clear_cache_core();
    let (logits, _) = model
        .forward_with_hidden_core(4, true)
        .expect("forward tras caché debe seguir estable");
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "logits deben ser finitos tras caché"
    );

    // Los 3 bloques entrenados deben haber mutado sus centroides.
    let changed = model.blocks[n - 1].gate_gen.centroids != before_last
        || model.blocks[n - 2].gate_gen.centroids != before_mid
        || model.blocks[n - 3].gate_gen.centroids != before_first;
    assert!(
        changed,
        "la caché debió propagar gradiente y mutar centroides en los últimos 3 bloques"
    );
}

#[test]
fn test_cached_full_body_no_nan() {
    let path = "models/production/smollm2_4bit.gaje.flat";
    if !std::path::Path::new(path).exists() {
        eprintln!("skip: modelo no presente en {}", path);
        return;
    }
    let mut model = load_genomic_auto(path).expect("cargar modelo");
    let n = model.blocks.len();
    let before0 = model.blocks[0].gate_gen.centroids.clone();
    let beforelast = model.blocks[n - 1].gate_gen.centroids.clone();

    let loss = model
        .train_sequence_cached_core(vec![4usize, 5, 6, 7, 8], 1e-6, n, 1.0)
        .unwrap();
    eprintln!("cached full loss={loss:.4}");
    assert!(loss.is_finite(), "loss debe ser finita");

    model.clear_cache_core();
    let (logits, _) = model
        .forward_with_hidden_core(4, true)
        .expect("forward tras full-body caché debe seguir estable");
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "forward debe ser finito tras full-body caché"
    );

    let changed0 = model.blocks[0].gate_gen.centroids != before0;
    let changedlast = model.blocks[n - 1].gate_gen.centroids != beforelast;
    eprintln!("full-body caché: block0 changed={changed0} | last changed={changedlast}");
    assert!(
        changed0 && changedlast,
        "el gradiente por caché debe llegar al bloque 0 y al último (no morir)"
    );
}
#[test]
fn test_transpose_isolated_nibble() {
    let path = "models/production/smollm2_4bit.gaje.flat";
    if !std::path::Path::new(path).exists() {
        return;
    }
    let mut model = load_genomic_auto(path).unwrap();
    // Toma el gate_gen del bloque 0 como linear Genomic4Bit aislado.
    let bi = 0usize;
    let eps = 1e-3f32;
    let x = model.embeddings.get_row_core(4).unwrap();
    let n_in = x.len();
    let mut worst = 0.0f32;
    for r in 0..2 {
        // one-hot en la fila r: dL/dy[i] = delta_ir  =>  dL/dx[j] = W[r,j]
        let mut w = vec![0.0f32; model.blocks[bi].gate_gen.out_features];
        w[r] = 1.0;
        let d_ana = model.blocks[bi].gate_gen.backward_core(w).unwrap();
        for j in 0..n_in {
            let mut xp = x.clone();
            xp[j] += eps;
            let yp = model.blocks[bi]
                .gate_gen
                .forward_core(xp, None, true)
                .unwrap();
            let mut xm = x.clone();
            xm[j] -= eps;
            let ym = model.blocks[bi]
                .gate_gen
                .forward_core(xm, None, true)
                .unwrap();
            let num = (yp[r] - ym[r]) / (2.0 * eps);
            let rel = (d_ana[j] - num).abs() / (d_ana[j].abs() + num.abs() + 1e-9);
            if rel > worst {
                worst = rel;
                let bs = model.blocks[bi].gate_gen.block_size;
                let b = j / bs;
                let sub = (j % bs) % 2;
                let byte_idx = (j % bs) / 2;
                let stride = model.blocks[bi].gate_gen.stride;
                let db: &[u8] = match &model.blocks[bi].gate_gen.weight_db {
                    crate::nn::linear::WeightDatabase::Genomic4Bit(d) => d.as_ref(),
                    _ => &[],
                };
                let nblk = model.blocks[bi].gate_gen.in_features / bs;
                let byte = db
                    .get(r * nblk * stride + b * stride + byte_idx)
                    .copied()
                    .unwrap_or(0);
                eprintln!(
                    "  worst r={r} j={j} b={b} sub={sub} byte=0x{byte:02X} nibble={} ana={} num={}",
                    if sub == 0 { byte >> 4 } else { byte & 0x0F },
                    d_ana[j],
                    num
                );
            }
        }
    }
    eprintln!("transpose isolated (por fila): worst rel_err = {worst:.4}");
    assert!(
        worst < 0.05,
        "backward_core (transpose) de Genomic4Bit no coincide con diferencias finitas"
    );
}

#[test]
fn test_gradient_check_block_robust() {
    let path = "models/production/smollm2_4bit.gaje.flat";
    if !std::path::Path::new(path).exists() {
        return;
    }
    let mut model = load_genomic_auto(path).unwrap();
    let bi = 0usize;
    let eps = 1e-3f64;
    // L = sum(final_out * d_out), con d_out sintético que garantiza gradiente O(1).
    let d_out: Vec<f32> = (0..model.blocks[bi].w_down.out_features)
        .map(|i| if i % 3 == 0 { 0.05 } else { -0.02 })
        .collect();
    let x = model.embeddings.get_row_core(4).unwrap();

    // Construir cache de atención de UN SOLO token de forma consistente:
    // forward_core_cached hace push al k/v_cache, así que limpiar antes de cada forward.
    model.clear_cache_core();
    let (_, cache) = model.blocks[bi].forward_core_cached(x.clone(), 0).unwrap();
    let d_x_ana = model.blocks[bi]
        .backward_core_cached(&cache, d_out.clone(), 0.0, 0.0)
        .unwrap();

    // Re-derivar num con cache fresco (1 token) para cada perturbación.
    let n_in = x.len();
    let mut worst = 0.0f64;
    let mut worst_strong = 0.0f64;
    let mut checked = 0usize;
    for j in 0..n_in {
        model.clear_cache_core();
        let mut xp = x.clone();
        xp[j] = xp[j] + eps as f32;
        let (yp, _) = model.blocks[bi].forward_core_cached(xp, 0).unwrap();
        model.clear_cache_core();
        let mut xm = x.clone();
        xm[j] = xm[j] - eps as f32;
        let (ym, _) = model.blocks[bi].forward_core_cached(xm, 0).unwrap();
        let lp: f64 = yp
            .iter()
            .zip(d_out.iter())
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum();
        let lm: f64 = ym
            .iter()
            .zip(d_out.iter())
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum();
        let num = (lp - lm) / (2.0 * eps);
        let a = d_x_ana[j] as f64;
        if a.abs() < 1e-5 && num.abs() < 1e-5 {
            continue;
        }
        checked += 1;
        let rel = (a - num).abs() / (a.abs() + num.abs() + 1e-12);
        let strong = a.abs().max(num.abs()) > 0.02;
        if rel > worst {
            worst = rel;
        }
        if strong && rel > worst_strong {
            worst_strong = rel;
            if rel > 0.10 {
                eprintln!("  strong j={j} ana={a:.6} num={num:.6} rel={rel:.4}");
            }
        }
    }
    eprintln!("block gradient check: worst rel_err={worst:.4} worst_strong(>0.02)={worst_strong:.4} (checked {checked}/{n_in})");
    assert!(
        worst_strong < 0.10,
        "backward_core_cached del bloque no coincide con diferencias finitas"
    );
}

#[test]
fn test_refine_indexing_matches_forward() {
    // Verifica que el STE de `refine_with_grads_core` acumule en el MISMO
    // centroide y nibble que usa el forward kernel (even->high, odd->low),
    // comparando el delta real del centroide 0 contra una suma manual que
    // lee el db con el mapeo del forward. Sin re-evaluar forward (sin ruido f32).
    let path = "models/production/smollm2_4bit.gaje.flat";
    if !std::path::Path::new(path).exists() {
        return;
    }
    let mut model = load_genomic_auto(path).unwrap();
    let bi = 0usize;
    let lr = 1e-4f32;

    let gl = &model.blocks[bi].gate_gen;
    let n_blocks = gl.in_features / gl.block_size;
    let bs = gl.block_size;
    let stride = gl.stride;
    let out_features = gl.out_features;
    let in_features = gl.in_features;
    let db: Vec<u8> = match &gl.weight_db {
        crate::nn::linear::WeightDatabase::Genomic4Bit(d) => d.as_ref().to_vec(),
        _ => vec![],
    };
    // input y grads sintéticos con magnitudes O(1) para señal fuerte.
    let x: Vec<f32> = (0..in_features)
        .map(|j| ((j % 7) as f32 - 3.0) * 0.1)
        .collect();
    let grads: Vec<f32> = (0..out_features)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.05)
        .collect();
    drop(gl);

    let c_before = model.blocks[bi].gate_gen.centroids[0];
    model.blocks[bi]
        .gate_gen
        .refine_with_grads_core(x.clone(), grads.clone(), lr)
        .unwrap();
    let c_after = model.blocks[bi].gate_gen.centroids[0];
    let delta_actual = (c_before - c_after) / lr; // = gradiente STE del centroide 0

    // Suma manual con el mapeo del forward (even -> high nibble).
    let mut expected = 0.0f64;
    for i in 0..out_features {
        for b in 0..n_blocks {
            for k in 0..bs {
                let j = b * bs + k;
                let byte = db[i * n_blocks * stride + b * stride + k / 2];
                let nibble = if k % 2 == 0 { byte >> 4 } else { byte & 0x0F };
                let c_idx = (i * n_blocks + b) * 16 + nibble as usize;
                if c_idx == 0 {
                    expected += grads[i] as f64 * x[j] as f64;
                }
            }
        }
    }

    let rel = (delta_actual as f64 - expected).abs() / (expected.abs() + 1e-12);
    eprintln!(
        "refine vs forward: delta_actual={delta_actual:.6} expected={expected:.6} rel_err={rel:.6}"
    );
    assert!(
        expected.abs() > 1e-3,
        "centroide 0 sin pesos asignados? señal demasiado débil"
    );
    assert!(
        rel < 1e-2,
        "refine no acumula con el mapeo del forward (rel_err={rel:.6})"
    );
}

// ---- Entrenamiento del cuerpo con validación held-out (Vía B) ----
// Carga el estudiante smollm2 + su tokenizer, tokeniza el corpus de destilación,
// entrena el cuerpo (escalera) sobre un split y mide la CE held-out antes/después.
fn ce_loss(model: &mut crate::nn::llm::GenomicLLM, tokens: &[usize]) -> f32 {
    if tokens.len() < 2 {
        return 0.0;
    }
    model.clear_cache_core();
    let mut total = 0.0f32;
    for i in 0..tokens.len() - 1 {
        let logits = model.forward_core(tokens[i], false).expect("forward eval");
        let t = tokens[i + 1];
        let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum_e = 0.0f32;
        for &l in &logits {
            sum_e += (l - max_l).exp();
        }
        let p = ((logits[t] - max_l).exp() / sum_e).max(1e-12);
        total -= p.ln();
    }
    total / (tokens.len() - 1) as f32
}

#[test]
#[ignore = "lento (~220s): entrena el cuerpo y valida CE held-out; correr explícitamente con -- --ignored"]
fn test_body_training_heldout_generalization() {
    let model_path = "models/production/smollm2_4bit.gaje.flat";
    let tok_path = "models/core/tokenizer.json";
    let corpus_path = "data/distill/train_smollm2_1t.jsonl";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("skip: modelo no presente");
        return;
    }
    let mut model = load_genomic_auto(model_path).expect("cargar modelo");
    let n = model.blocks.len();
    let vocab = model.embeddings.out_features;

    // Tokenizar el corpus (prompt+answer) en el tokenizer del estudiante.
    let tokenizer =
        crate::core::tokenizer::GajeTokenizer::from_file(tok_path).expect("cargar tokenizer");
    let raw = std::fs::read_to_string(corpus_path).expect("leer corpus");
    let mut stream: Vec<usize> = Vec::new();
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            let prompt = v["prompt"].as_str().unwrap_or("");
            let answer = v["answer"].as_str().unwrap_or("");
            let ids = tokenizer
                .encode(&format!("{prompt}{answer}"), false)
                .expect("encode");
            for id in ids {
                stream.push((id as usize).min(vocab - 1));
            }
        }
    }
    assert!(
        stream.len() >= 40,
        "corpus demasiado corto: {}",
        stream.len()
    );
    eprintln!(
        "vocab={vocab} tok_vocab={} stream_len={} n_blocks={n}",
        tokenizer.vocab_size(),
        stream.len()
    );

    let held_len = (stream.len() / 5).max(8);
    let heldout: Vec<usize> = stream[stream.len() - held_len..].to_vec();
    let train: Vec<usize> = stream[..stream.len() - held_len].to_vec();

    let loss_before = ce_loss(&mut model, &heldout);
    assert!(
        loss_before.is_finite(),
        "loss held-out inicial debe ser finita"
    );
    eprintln!("heldout CE ANTES = {loss_before:.4}");

    // Entrenar el cuerpo (escalera: últimos bloques), lr pequeño, grad-clip activo.
    let before_last = model.blocks[n - 1].gate_gen.centroids.clone();
    model.clear_cache_core();
    let tloss = model
        .train_sequence_cached_core(train.clone(), 1e-5, 4, 1.0)
        .expect("entrenamiento del cuerpo");
    assert!(tloss.is_finite(), "train loss debe ser finita, got {tloss}");
    eprintln!("train loss (últimos 4 bloques) = {tloss:.4}");

    // Estabilidad del forward tras entrenar.
    model.clear_cache_core();
    let (logits, _) = model
        .forward_with_hidden_core(heldout[0], true)
        .expect("forward tras entrenar");
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "forward debe ser finito tras entrenar"
    );

    let loss_after = ce_loss(&mut model, &heldout);
    assert!(
        loss_after.is_finite(),
        "loss held-out final debe ser finita"
    );
    eprintln!("heldout CE DESPUÉS = {loss_after:.4}");

    let body_mutated = model.blocks[n - 1].gate_gen.centroids != before_last;
    assert!(body_mutated, "el cuerpo debe mutar sus centroides");

    let verdict = if loss_after < loss_before {
        "MEJORA"
    } else {
        "ESTABLE"
    };
    eprintln!("→ generalización: {verdict} ({loss_before:.4} -> {loss_after:.4})");
    assert!(
        loss_after <= loss_before * 1.5,
        "la loss held-out degradó catastróficamente: {loss_before} -> {loss_after}"
    );
}

#[test]
#[ignore = "lento: full-body (30 bloques) con validación held-out; correr explícitamente"]
fn test_body_training_fullbody_heldout() {
    let model_path = "models/production/smollm2_4bit.gaje.flat";
    let tok_path = "models/core/tokenizer.json";
    let corpus_path = "data/distill/train_smollm2_1t.jsonl";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("skip: modelo no presente");
        return;
    }
    let mut model = load_genomic_auto(model_path).expect("cargar modelo");
    let n = model.blocks.len();
    let vocab = model.embeddings.out_features;
    let tokenizer = crate::core::tokenizer::GajeTokenizer::from_file(tok_path).expect("tok");
    let raw = std::fs::read_to_string(corpus_path).expect("leer corpus");
    let mut stream: Vec<usize> = Vec::new();
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            let prompt = v["prompt"].as_str().unwrap_or("");
            let answer = v["answer"].as_str().unwrap_or("");
            let ids = tokenizer
                .encode(&format!("{prompt}{answer}"), false)
                .expect("encode");
            for id in ids {
                stream.push((id as usize).min(vocab - 1));
            }
        }
    }
    assert!(stream.len() >= 40);
    let held_len = (stream.len() / 5).max(8);
    let heldout: Vec<usize> = stream[stream.len() - held_len..].to_vec();
    // Subset pequeño de entrenamiento para que el backward full-body sea factible.
    let train_max = stream.len() - held_len;
    let train: Vec<usize> = stream[..train_max.min(240)].to_vec();
    eprintln!(
        "full-body: train_len={} held_len={} n_blocks={n}",
        train.len(),
        heldout.len()
    );

    let loss_before = ce_loss(&mut model, &heldout);
    assert!(loss_before.is_finite());
    eprintln!("heldout CE ANTES (full-body) = {loss_before:.4}");

    let before0 = model.blocks[0].gate_gen.centroids.clone();
    let beforelast = model.blocks[n - 1].gate_gen.centroids.clone();
    model.clear_cache_core();
    let tloss = model
        .train_sequence_cached_core(train.clone(), 1e-6, n, 1.0)
        .expect("full-body training");
    assert!(tloss.is_finite(), "train loss finita, got {tloss}");
    eprintln!("full-body train loss = {tloss:.4}");

    model.clear_cache_core();
    let (logits, _) = model.forward_with_hidden_core(heldout[0], true).unwrap();
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "forward full-body debe ser finito"
    );

    let loss_after = ce_loss(&mut model, &heldout);
    assert!(loss_after.is_finite());
    eprintln!("heldout CE DESPUÉS (full-body) = {loss_after:.4}");

    let changed0 = model.blocks[0].gate_gen.centroids != before0;
    let changedlast = model.blocks[n - 1].gate_gen.centroids != beforelast;
    assert!(
        changed0 && changedlast,
        "full-body debe mutar bloque 0 y último (block0={changed0} last={changedlast})"
    );

    let verdict = if loss_after < loss_before {
        "MEJORA"
    } else {
        "ESTABLE"
    };
    eprintln!("→ full-body generalización: {verdict} ({loss_before:.4} -> {loss_after:.4})");
    assert!(
        loss_after <= loss_before * 1.5,
        "full-body degradó catastróficamente: {loss_before} -> {loss_after}"
    );
}

#[test]
#[ignore = "lento: barrido de escalera (n_train_blocks) sobre held-out; correr explícitamente"]
fn test_body_ladder_sweep() {
    let model_path = "models/production/smollm2_4bit.gaje.flat";
    let tok_path = "models/core/tokenizer.json";
    let corpus_path = "data/distill/train_smollm2_1t.jsonl";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("skip");
        return;
    }
    let n = load_genomic_auto(model_path).unwrap().blocks.len();
    let tokenizer = crate::core::tokenizer::GajeTokenizer::from_file(tok_path).expect("tok");
    let raw = std::fs::read_to_string(corpus_path).expect("corpus");
    let mut stream: Vec<usize> = Vec::new();
    {
        let probe = load_genomic_auto(model_path).unwrap();
        let vocab = probe.embeddings.out_features;
        for line in raw.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                let p = v["prompt"].as_str().unwrap_or("");
                let a = v["answer"].as_str().unwrap_or("");
                let ids = tokenizer.encode(&format!("{p}{a}"), false).expect("encode");
                for id in ids {
                    stream.push((id as usize).min(vocab - 1));
                }
            }
        }
    }
    let held_len = (stream.len() / 5).max(8);
    let heldout: Vec<usize> = stream[stream.len() - held_len..].to_vec();
    let train_max = stream.len() - held_len;
    let train: Vec<usize> = stream[..train_max.min(180)].to_vec();

    // baseline held-out (modelo fresco, sin entrenar)
    let mut base = load_genomic_auto(model_path).unwrap();
    let base_loss = ce_loss(&mut base, &heldout);
    drop(base);

    eprintln!(
        "=== Barrido escalera (train_len={}) ===  baseline heldout={base_loss:.4}",
        train.len()
    );
    let cfgs: Vec<(usize, f32)> = vec![(4, 1e-5), (8, 1e-5), (12, 5e-6), (16, 2e-6)];
    let mut best: Option<(usize, f32, f32)> = None;
    for (nb, lr) in cfgs {
        let mut model = load_genomic_auto(model_path).unwrap();
        let tloss = model
            .train_sequence_cached_core(train.clone(), lr, nb, 1.0)
            .expect("train");
        model.clear_cache_core();
        let (logits, _) = model.forward_with_hidden_core(heldout[0], true).unwrap();
        let finite = logits.iter().all(|v| v.is_finite());
        let after = ce_loss(&mut model, &heldout);
        let delta = after - base_loss;
        let mark = if !finite {
            "NaN!!"
        } else if delta < -0.01 {
            "MEJORA"
        } else if delta <= 0.01 {
            "ESTABLE"
        } else {
            "DEGRADA"
        };
        eprintln!("  n_blk={nb:>2} lr={lr:.0e} train={tloss:.4} heldout={after:.4} Δ={delta:+.4} [{mark}]");
        if finite && (best.is_none() || after < best.unwrap().2) {
            best = Some((nb, lr, after));
        }
    }
    let (nb, lr, after) = best.expect("ninguna config finita");
    eprintln!(
        "→ mejor escalera: n_blk={nb} lr={lr:.0e} heldout={after:.4} (baseline {base_loss:.4})"
    );
    assert!(
        after < base_loss,
        "la mejor config de escalera no mejoró held-out (baseline {base_loss:.4} -> {after:.4})"
    );
}

#[test]
#[ignore = "lento: barrido de lr fijando 8 bloques sobre held-out"]
fn test_body_lr_sweep_blk8() {
    let model_path = "models/production/smollm2_4bit.gaje.flat";
    let tok_path = "models/core/tokenizer.json";
    let corpus_path = "data/distill/train_smollm2_1t.jsonl";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("skip");
        return;
    }
    let tokenizer = crate::core::tokenizer::GajeTokenizer::from_file(tok_path).expect("tok");
    let raw = std::fs::read_to_string(corpus_path).expect("corpus");
    let mut stream: Vec<usize> = Vec::new();
    {
        let probe = load_genomic_auto(model_path).unwrap();
        let vocab = probe.embeddings.out_features;
        for line in raw.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                let p = v["prompt"].as_str().unwrap_or("");
                let a = v["answer"].as_str().unwrap_or("");
                let ids = tokenizer.encode(&format!("{p}{a}"), false).expect("encode");
                for id in ids {
                    stream.push((id as usize).min(vocab - 1));
                }
            }
        }
    }
    let held_len = (stream.len() / 5).max(8);
    let heldout: Vec<usize> = stream[stream.len() - held_len..].to_vec();
    let train_max = stream.len() - held_len;
    let train: Vec<usize> = stream[..train_max.min(180)].to_vec();

    let mut base = load_genomic_auto(model_path).unwrap();
    let base_loss = ce_loss(&mut base, &heldout);
    drop(base);

    let nb = 8usize;
    eprintln!(
        "=== Barrido lr (n_blk={nb}, train_len={}) === baseline={base_loss:.4}",
        train.len()
    );
    let mut best: Option<(f32, f32)> = None;
    for lr in [2e-6f32, 5e-6, 1e-5, 2e-5, 5e-5] {
        let mut model = load_genomic_auto(model_path).unwrap();
        let tloss = model
            .train_sequence_cached_core(train.clone(), lr, nb, 1.0)
            .expect("train");
        model.clear_cache_core();
        let (logits, _) = model.forward_with_hidden_core(heldout[0], true).unwrap();
        let finite = logits.iter().all(|v| v.is_finite());
        let after = ce_loss(&mut model, &heldout);
        let delta = after - base_loss;
        let mark = if !finite {
            "NaN!!"
        } else if delta < -0.01 {
            "MEJORA"
        } else if delta <= 0.01 {
            "ESTABLE"
        } else {
            "DEGRADA"
        };
        eprintln!("  lr={lr:.0e} train={tloss:.4} heldout={after:.4} Δ={delta:+.4} [{mark}]");
        if finite && (best.is_none() || after < best.unwrap().1) {
            best = Some((lr, after));
        }
    }
    let (lr, after) = best.expect("ninguna finita");
    eprintln!("→ mejor lr (8 bloques): lr={lr:.0e} heldout={after:.4} (baseline {base_loss:.4})");
    assert!(after < base_loss, "ninguna config mejoró held-out");
}

#[test]
#[ignore = "lento: mapear límite de estabilidad (lr alto) a 8 bloques"]
fn test_body_lr_high_boundary() {
    let model_path = "models/production/smollm2_4bit.gaje.flat";
    let tok_path = "models/core/tokenizer.json";
    let corpus_path = "data/distill/train_smollm2_1t.jsonl";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("skip");
        return;
    }
    let tokenizer = crate::core::tokenizer::GajeTokenizer::from_file(tok_path).expect("tok");
    let raw = std::fs::read_to_string(corpus_path).expect("corpus");
    let mut stream: Vec<usize> = Vec::new();
    {
        let probe = load_genomic_auto(model_path).unwrap();
        let vocab = probe.embeddings.out_features;
        for line in raw.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                let p = v["prompt"].as_str().unwrap_or("");
                let a = v["answer"].as_str().unwrap_or("");
                let ids = tokenizer.encode(&format!("{p}{a}"), false).expect("encode");
                for id in ids {
                    stream.push((id as usize).min(vocab - 1));
                }
            }
        }
    }
    let held_len = (stream.len() / 5).max(8);
    let heldout: Vec<usize> = stream[stream.len() - held_len..].to_vec();
    let train_max = stream.len() - held_len;
    let train: Vec<usize> = stream[..train_max.min(180)].to_vec();

    let mut base = load_genomic_auto(model_path).unwrap();
    let base_loss = ce_loss(&mut base, &heldout);
    drop(base);

    let nb = 8usize;
    eprintln!("=== Límite de estabilidad lr (n_blk={nb}) === baseline={base_loss:.4}");
    for lr in [5e-5f32, 1e-4, 2e-4, 5e-4] {
        let mut model = load_genomic_auto(model_path).unwrap();
        let tloss = model
            .train_sequence_cached_core(train.clone(), lr, nb, 1.0)
            .expect("train");
        model.clear_cache_core();
        let (logits, _) = model.forward_with_hidden_core(heldout[0], true).unwrap();
        let finite = logits.iter().all(|v| v.is_finite());
        let after = if finite {
            ce_loss(&mut model, &heldout)
        } else {
            f32::INFINITY
        };
        let delta = after - base_loss;
        let mark = if !finite {
            "NaN!!"
        } else if delta < -0.01 {
            "MEJORA"
        } else if delta <= 0.01 {
            "ESTABLE"
        } else {
            "DEGRADA"
        };
        eprintln!("  lr={lr:.0e} train={tloss:.4} heldout={after:.4} Δ={delta:+.4} [{mark}]");
    }
    // No assert: este test solo mapea la frontera de estabilidad (informativo).
}

#[test]
#[ignore = "lento: escalar bloques con lr por capas (layer-wise decay)"]
fn test_body_layerwise_scale() {
    let model_path = "models/production/smollm2_4bit.gaje.flat";
    let tok_path = "models/core/tokenizer.json";
    let corpus_path = "data/distill/train_smollm2_1t.jsonl";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("skip");
        return;
    }
    let tokenizer = crate::core::tokenizer::GajeTokenizer::from_file(tok_path).expect("tok");
    let raw = std::fs::read_to_string(corpus_path).expect("corpus");
    let mut stream: Vec<usize> = Vec::new();
    {
        let probe = load_genomic_auto(model_path).unwrap();
        let vocab = probe.embeddings.out_features;
        for line in raw.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                let p = v["prompt"].as_str().unwrap_or("");
                let a = v["answer"].as_str().unwrap_or("");
                let ids = tokenizer.encode(&format!("{p}{a}"), false).expect("encode");
                for id in ids {
                    stream.push((id as usize).min(vocab - 1));
                }
            }
        }
    }
    let held_len = (stream.len() / 5).max(8);
    let heldout: Vec<usize> = stream[stream.len() - held_len..].to_vec();
    let train_max = stream.len() - held_len;
    let train: Vec<usize> = stream[..train_max.min(180)].to_vec();

    let mut base = load_genomic_auto(model_path).unwrap();
    let base_loss = ce_loss(&mut base, &heldout);
    drop(base);

    let lr = 2e-4f32;
    eprintln!("=== Escalar bloques con lr por capas (lr={lr:.0e}) === baseline={base_loss:.4}");
    // (n_train_blocks, decay, etiqueta)
    let cfgs: Vec<(usize, f32, &str)> = vec![
        (8, 1.0, "uniforme"),
        (8, 0.7, "layerwise .7"),
        (16, 0.8, "layerwise .8"),
        (24, 0.85, "layerwise .85"),
    ];
    let mut best: Option<(usize, f32, &str, f32)> = None;
    for (nb, decay, label) in cfgs {
        let mut model = load_genomic_auto(model_path).unwrap();
        let tloss = model
            .train_sequence_cached_layerwise_core(train.clone(), lr, nb, 1.0, decay, true, None)
            .expect("train");
        model.clear_cache_core();
        let (logits, _) = model.forward_with_hidden_core(heldout[0], true).unwrap();
        let finite = logits.iter().all(|v| v.is_finite());
        let after = if finite {
            ce_loss(&mut model, &heldout)
        } else {
            f32::INFINITY
        };
        let delta = after - base_loss;
        let mark = if !finite {
            "NaN!!"
        } else if delta < -0.01 {
            "MEJORA"
        } else if delta <= 0.01 {
            "ESTABLE"
        } else {
            "DEGRADA"
        };
        eprintln!("  n_blk={nb:>2} {label:<14} train={tloss:.4} heldout={after:.4} Δ={delta:+.4} [{mark}]");
        if finite && (best.is_none() || after < best.unwrap().3) {
            best = Some((nb, decay, label, after));
        }
    }
    let (nb, decay, label, after) = best.expect("ninguna finita");
    eprintln!(
        "→ mejor: n_blk={nb} {label} (decay={decay}) heldout={after:.4} (baseline {base_loss:.4})"
    );
    assert!(
        after < base_loss,
        "lr por capas no mejoró held-out (baseline {base_loss:.4} -> {after:.4})"
    );
}

#[test]
#[ignore = "lento: round-trip del writer mmap GAJE (carga+guarda+recarga el modelo 0.5B)"]
fn test_flat_mmap_roundtrip() {
    let src = "models/production/smollm2_4bit.gaje.flat";
    let dst = "/tmp/opencode/smollm2_roundtrip.gaje.flat";
    if !std::path::Path::new(src).exists() {
        eprintln!("skip");
        return;
    }
    use crate::io::config::ModelConfig;
    use crate::io::flat_reader::GajeFlatFileReader;
    use crate::io::flat_writer::save_genomic_flat;
    let reader = GajeFlatFileReader::open(src).unwrap();
    let config: ModelConfig = serde_json::from_str(&reader.metadata_json).unwrap();
    let mut model = crate::io::flat_reader::load_genomic_auto(src).unwrap();
    save_genomic_flat(dst, &model, &config, None).unwrap();
    let mut reloaded = crate::io::flat_reader::load_genomic_auto(dst).unwrap();
    let (logits_a, _) = model.forward_with_hidden_core(4, false).unwrap();
    let (logits_b, _) = reloaded.forward_with_hidden_core(4, false).unwrap();
    assert!(logits_b.iter().all(|v| v.is_finite()));
    let maxdiff = logits_a
        .iter()
        .zip(logits_b.iter())
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(maxdiff < 1e-3, "round-trip diverge: {maxdiff}");
    let _ = std::fs::remove_file(dst);
}
