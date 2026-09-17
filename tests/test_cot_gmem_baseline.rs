#[cfg(test)]
mod tests {
    use _impl::io::flat_reader::GajeFlatFileReader;
    use _impl::io::gmem::GmemMemoryIndex;
    use _impl::nn::llm::GenomicLLM;
    use _impl::core::gtok::GtokNativeTokenizer;
    use std::time::Instant;

    #[derive(Clone)]
    struct PromptTestCase {
        id: usize,
        domain: &'static str,
        question: &'static str,
        fact: &'static str,
        query_term: &'static str,
        correct_pattern: &'static str,
        distractor_pattern: Option<&'static str>,
    }

    fn get_test_cases() -> Vec<PromptTestCase> {
        vec![
            // 🌍 Geografía y Capitales (5 prompts)
            PromptTestCase {
                id: 1,
                domain: "Geography",
                question: "What is the capital of Australia?",
                fact: "The capital of Australia is Canberra.",
                query_term: "capital of Australia",
                correct_pattern: r"(?i)\bcanberra\b",
                distractor_pattern: Some(r"(?i)\bsydney\b|\bmelbourne\b"),
            },
            PromptTestCase {
                id: 2,
                domain: "Geography",
                question: "What is the capital of Canada?",
                fact: "The capital of Canada is Ottawa.",
                query_term: "capital of Canada",
                correct_pattern: r"(?i)\bottawa\b",
                distractor_pattern: Some(r"(?i)\btoronto\b|\bmontreal\b"),
            },
            PromptTestCase {
                id: 3,
                domain: "Geography",
                question: "What is the capital of Brazil?",
                fact: "The capital of Brazil is Brasilia.",
                query_term: "capital of Brazil",
                correct_pattern: r"(?i)\bbrasilia\b",
                distractor_pattern: Some(r"(?i)\brio\b|\bsao paulo\b"),
            },
            PromptTestCase {
                id: 4,
                domain: "Geography",
                question: "What is the capital of Turkey?",
                fact: "The capital of Turkey is Ankara.",
                query_term: "capital of Turkey",
                correct_pattern: r"(?i)\bankara\b",
                distractor_pattern: Some(r"(?i)\bistanbul\b"),
            },
            PromptTestCase {
                id: 5,
                domain: "Geography",
                question: "What is the capital of Switzerland?",
                fact: "The capital of Switzerland is Bern.",
                query_term: "capital of Switzerland",
                correct_pattern: r"(?i)\bbern\b",
                distractor_pattern: Some(r"(?i)\bzurich\b|\bgeneva\b"),
            },

            // 🔬 Ciencia y Constantes Físicas (5 prompts)
            PromptTestCase {
                id: 6,
                domain: "Science",
                question: "What is the chemical symbol for Gold?",
                fact: "The chemical symbol for gold is Au.",
                query_term: "chemical symbol for Gold",
                correct_pattern: r"(?i)\bAu\b|gold\s+is\s+Au",
                distractor_pattern: Some(r"(?i)\bAg\b|\bFe\b|\bCu\b"),
            },
            PromptTestCase {
                id: 7,
                domain: "Science",
                question: "What is the chemical symbol for Lead?",
                fact: "The chemical symbol for lead is Pb.",
                query_term: "chemical symbol for Lead",
                correct_pattern: r"(?i)\bPb\b|lead\s+is\s+Pb",
                distractor_pattern: Some(r"(?i)\bLd\b|\bLe\b|\bFe\b"),
            },
            PromptTestCase {
                id: 8,
                domain: "Science",
                question: "What is the escape velocity of Earth in km/s?",
                fact: "The escape velocity of Earth is 11.2 km/s.",
                query_term: "escape velocity of Earth",
                correct_pattern: r"(?i)11[\.,]?2|11[\.,]186",
                distractor_pattern: Some(r"(?i)\b16[\.,]8|\b9[\.,]8\b|\b7[\.,]9\b"),
            },
            PromptTestCase {
                id: 9,
                domain: "Science",
                question: "What is the speed of light in vacuum in km/s?",
                fact: "The speed of light in vacuum is approximately 299,792 km/s.",
                query_term: "speed of light in vacuum",
                correct_pattern: r"(?i)299[\.,]?792|300[\.,]?000|3\s*[\*x×]\s*10\^?8|300\s*thousand",
                distractor_pattern: Some(r"(?i)\b150[\.,]?000\b"),
            },
            PromptTestCase {
                id: 10,
                domain: "Science",
                question: "What is the most abundant gas in Earth's atmosphere?",
                fact: "The most abundant gas in Earth's atmosphere is Nitrogen (78%).",
                query_term: "most abundant gas in Earth atmosphere",
                correct_pattern: r"(?i)\bnitrogen\b|N2",
                distractor_pattern: Some(r"(?i)\boxygen\b|\bcarbon\b|\bargon\b"),
            },

            // 🪐 Astronomía y Sistema Solar (5 prompts)
            PromptTestCase {
                id: 11,
                domain: "Astronomy",
                question: "Which planet has the most moons in our solar system?",
                fact: "Saturn has the most confirmed moons in the solar system (146 moons).",
                query_term: "planet with most moons",
                correct_pattern: r"(?i)\bsaturn\b|\bjupiter\b",
                distractor_pattern: Some(r"(?i)\bearth\b|\bmars\b|\bneptune\b|\buranus\b"),
            },
            PromptTestCase {
                id: 12,
                domain: "Astronomy",
                question: "What is the closest planet to the Sun?",
                fact: "Mercury is the closest planet to the Sun.",
                query_term: "closest planet to the Sun",
                correct_pattern: r"(?i)\bmercury\b",
                distractor_pattern: Some(r"(?i)\bvenus\b|\bmars\b"),
            },
            PromptTestCase {
                id: 13,
                domain: "Astronomy",
                question: "What is the largest moon of Jupiter?",
                fact: "Ganymede is the largest moon of Jupiter.",
                query_term: "largest moon of Jupiter",
                correct_pattern: r"(?i)\bganymede\b",
                distractor_pattern: Some(r"(?i)\bio\b|\beuropa\b|\bcallisto\b|\btitan\b"),
            },
            PromptTestCase {
                id: 14,
                domain: "Astronomy",
                question: "What is the second planet from the Sun?",
                fact: "Venus is the second planet from the Sun.",
                query_term: "second planet from the Sun",
                correct_pattern: r"(?i)\bvenus\b",
                distractor_pattern: Some(r"(?i)\bmercury\b|\bearth\b|\bmars\b"),
            },
            PromptTestCase {
                id: 15,
                domain: "Astronomy",
                question: "Which planet is known as the Morning Star?",
                fact: "Venus is the planet commonly known as the Morning Star.",
                query_term: "planet known as Morning Star",
                correct_pattern: r"(?i)\bvenus\b",
                distractor_pattern: Some(r"(?i)\bmars\b|\bmercury\b|\bjupiter\b"),
            },

            // 📜 Historia y Números Canónicos (5 prompts)
            PromptTestCase {
                id: 16,
                domain: "History",
                question: "In what year did the Apollo 11 moon landing occur?",
                fact: "The Apollo 11 moon landing occurred in 1969.",
                query_term: "year of Apollo 11 moon landing",
                correct_pattern: r"\b1969\b",
                distractor_pattern: Some(r"\b2011\b|\b197\d\b|\b196[0-8]\b"),
            },
            PromptTestCase {
                id: 17,
                domain: "History",
                question: "In what year did World War II end?",
                fact: "World War II ended in 1945.",
                query_term: "year World War II ended",
                correct_pattern: r"\b1945\b",
                distractor_pattern: Some(r"\b2019\b|\b194[0-4]\b|\b1939\b|\b1918\b"),
            },
            PromptTestCase {
                id: 18,
                domain: "History",
                question: "In what year was the United Nations founded?",
                fact: "The United Nations was founded in 1945.",
                query_term: "year United Nations was founded",
                correct_pattern: r"\b1945\b|\b1942\b",
                distractor_pattern: Some(r"\b1919\b|\b1920\b"),
            },
            PromptTestCase {
                id: 19,
                domain: "Science",
                question: "How many elements are in the periodic table?",
                fact: "There are 118 confirmed elements in the periodic table.",
                query_term: "elements in periodic table",
                correct_pattern: r"\b118\b",
                distractor_pattern: Some(r"\b11[0-7]\b|\b10\d\b|\b92\b"),
            },
            PromptTestCase {
                id: 20,
                domain: "Biology",
                question: "How many chromosomes do human somatic cells have?",
                fact: "Human somatic cells have 46 chromosomes (23 pairs).",
                query_term: "chromosomes in human somatic cells",
                correct_pattern: r"\b46\b|23\s*pairs",
                distractor_pattern: Some(r"\b23\b|\b48\b|\b44\b"),
            },
        ]
    }

    fn check_answer(response_text: &str, correct_pattern: &str, distractor_pattern: Option<&str>) -> bool {
        let re_correct = regex::Regex::new(correct_pattern).unwrap();
        if let Some(correct_mat) = re_correct.find(response_text) {
            let correct_pos = correct_mat.start();
            if let Some(dist_pat) = distractor_pattern {
                let re_dist = regex::Regex::new(dist_pat).unwrap();
                if let Some(dist_mat) = re_dist.find(response_text) {
                    if dist_mat.start() < correct_pos {
                        // Distractor apareció antes que la respuesta correcta -> FALLO
                        return false;
                    }
                }
            }
            return true;
        }
        false
    }

    /// CONDICIÓN A: Baseline Directo sin CoT ni Memoria Externa
    /// Prompt estándar: <|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n
    #[test]
    fn test_condition_a_baseline() {
        println!("\n========================================================");
        println!("🎯 EVALUACIÓN EMPÍRICA — CONDICIÓN A: BASELINE DIRECTO");
        println!("Modelo: Qwen2.5-0.5B-Instruct (models/production/qwen2_5_0_5b.gaje) [Q4_0]");
        println!("========================================================\n");

        let test_cases = get_test_cases();
        let path = "models/production/qwen2_5_0_5b.gaje";
        let reader = GajeFlatFileReader::open(path).expect("Open Qwen Q4_0");
        let tokenizer = reader.get_embedded_gtok().expect("Get GTOK");
        let mut model = reader.load_genomic().expect("Load GenomicLLM");
        let stop_tokens = vec![151645, 151643]; // <|im_end|>, <|endoftext|>

        let mut correct_count = 0;
        let total_start = Instant::now();

        for tc in &test_cases {
            let chat_prompt = format!(
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                tc.question
            );
            let input_ids: Vec<usize> = tokenizer.encode(&chat_prompt).into_iter().map(|t| t as usize).collect();

            let output_ids = model.generate_native_core(input_ids, 25, 0.0, 1.15, stop_tokens.clone())
                .expect("Generation failed");

            let output_u32: Vec<u32> = output_ids.iter().map(|&t| t as u32).collect();
            let response_text = tokenizer.decode(&output_u32);

            let is_hit = check_answer(&response_text, tc.correct_pattern, tc.distractor_pattern);
            if is_hit {
                correct_count += 1;
            }

            let status_emoji = if is_hit { "✅ ACIERTO" } else { "❌ FALLO  " };
            println!("[{:02}/20] {} [{}] Q: {}", tc.id, status_emoji, tc.domain, tc.question);
            println!("       A Output: \"{}\"", response_text.trim().replace('\n', " "));
        }

        let elapsed = total_start.elapsed();
        let accuracy_pct = (correct_count as f32 / test_cases.len() as f32) * 100.0;

        println!("\n========================================================");
        println!("📊 RESUMEN FINAL — CONDICIÓN A (BASELINE DIRECTO)");
        println!("========================================================");
        println!("Aciertos Válidos:            {} / {}", correct_count, test_cases.len());
        println!("Exactitud Condición A:       {:.1}%", accuracy_pct);
        println!("Tiempo Total:                {:.2?}", elapsed);
        println!("========================================================\n");
    }

    /// CONDICIÓN B: CoT Libre sin Memoria Externa
    /// Prompt: Pide explícitamente "think step by step" y prefija la respuesta con "Thinking Process: "
    #[test]
    fn test_condition_b_cot_free() {
        println!("\n========================================================");
        println!("🧠 EVALUACIÓN EMPÍRICA — CONDICIÓN B: CoT LIBRE SIN MEMORIA");
        println!("Modelo: Qwen2.5-0.5B-Instruct (models/production/qwen2_5_0_5b.gaje) [Q4_0]");
        println!("========================================================\n");

        let test_cases = get_test_cases();
        let path = "models/production/qwen2_5_0_5b.gaje";
        let reader = GajeFlatFileReader::open(path).expect("Open Qwen Q4_0");
        let tokenizer = reader.get_embedded_gtok().expect("Get GTOK");
        let mut model = reader.load_genomic().expect("Load GenomicLLM");
        let stop_tokens = vec![151645, 151643]; // <|im_end|>, <|endoftext|>

        let mut correct_count = 0;
        let mut spontaneous_query_count = 0;
        let total_start = Instant::now();

        for tc in &test_cases {
            let chat_prompt = format!(
                "<|im_start|>user\n{} Please think step by step before answering.<|im_end|>\n<|im_start|>assistant\nThinking Process: ",
                tc.question
            );
            let input_ids: Vec<usize> = tokenizer.encode(&chat_prompt).into_iter().map(|t| t as usize).collect();

            // max_tokens = 60 para permitir que el razonamiento se desarrolle y emita conclusión
            let output_ids = model.generate_native_core(input_ids, 60, 0.0, 1.15, stop_tokens.clone())
                .expect("Generation failed");

            let output_u32: Vec<u32> = output_ids.iter().map(|&t| t as u32).collect();
            let response_text = tokenizer.decode(&output_u32);

            if response_text.to_lowercase().contains("[query:") || response_text.to_lowercase().contains("query:") {
                spontaneous_query_count += 1;
            }

            let is_hit = check_answer(&response_text, tc.correct_pattern, tc.distractor_pattern);
            if is_hit {
                correct_count += 1;
            }

            let status_emoji = if is_hit { "✅ ACIERTO" } else { "❌ FALLO  " };
            println!("[{:02}/20] {} [{}] Q: {}", tc.id, status_emoji, tc.domain, tc.question);
            println!("       CoT Output: \"{}\"", response_text.trim().replace('\n', " "));
        }

        let elapsed = total_start.elapsed();
        let accuracy_pct = (correct_count as f32 / test_cases.len() as f32) * 100.0;

        println!("\n========================================================");
        println!("📊 RESUMEN FINAL — CONDICIÓN B (CoT LIBRE SIN MEMORIA)");
        println!("========================================================");
        println!("Aciertos Válidos:            {} / {}", correct_count, test_cases.len());
        println!("Exactitud Condición B:       {:.1}% (Baseline A fue 30.0%)", accuracy_pct);
        println!("Queries Espontáneas [QUERY]: {} / 20", spontaneous_query_count);
        println!("Tiempo Total:                {:.2?}", elapsed);
        println!("========================================================\n");
    }

    /// CONDICIÓN C: CoT + Hecho Inyectado Forzado
    /// Scaffolding: Inyecta el recuerdo fáctico dentro de la cadena [THINKING] [QUERY: ...] -> [Hecho]
    #[test]
    fn test_condition_c_cot_forced_fact() {
        println!("\n========================================================");
        println!("🧬 EVALUACIÓN EMPÍRICA — CONDICIÓN C: CoT + HECHO FORZADO");
        println!("Modelo: Qwen2.5-0.5B-Instruct (models/production/qwen2_5_0_5b.gaje) [Q4_0]");
        println!("========================================================\n");

        let test_cases = get_test_cases();
        let path = "models/production/qwen2_5_0_5b.gaje";
        let reader = GajeFlatFileReader::open(path).expect("Open Qwen Q4_0");
        let tokenizer = reader.get_embedded_gtok().expect("Get GTOK");
        let mut model = reader.load_genomic().expect("Load GenomicLLM");
        let stop_tokens = vec![151645, 151643];

        let mut correct_count = 0;
        let total_start = Instant::now();

        for tc in &test_cases {
            let chat_prompt = format!(
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n[THINKING] [QUERY: {}] -> {}\n[ANSWER] ",
                tc.question, tc.query_term, tc.fact
            );
            let input_ids: Vec<usize> = tokenizer.encode(&chat_prompt).into_iter().map(|t| t as usize).collect();

            let output_ids = model.generate_native_core(input_ids, 25, 0.0, 1.15, stop_tokens.clone())
                .expect("Generation failed");

            let output_u32: Vec<u32> = output_ids.iter().map(|&t| t as u32).collect();
            let response_text = tokenizer.decode(&output_u32);

            let is_hit = check_answer(&response_text, tc.correct_pattern, tc.distractor_pattern);
            if is_hit {
                correct_count += 1;
            }

            let status_emoji = if is_hit { "✅ ACIERTO" } else { "❌ FALLO  " };
            println!("[{:02}/20] {} [{}] Q: {}", tc.id, status_emoji, tc.domain, tc.question);
            println!("       C Output: \"{}\"", response_text.trim().replace('\n', " "));
        }

        let elapsed = total_start.elapsed();
        let accuracy_pct = (correct_count as f32 / test_cases.len() as f32) * 100.0;

        println!("\n========================================================");
        println!("📊 RESUMEN FINAL — CONDICIÓN C (CoT + HECHO FORZADO)");
        println!("========================================================");
        println!("Aciertos Válidos:            {} / {}", correct_count, test_cases.len());
        println!("Exactitud Condición C:       {:.1}%", accuracy_pct);
        println!("Tiempo Total:                {:.2?}", elapsed);
        println!("========================================================\n");
    }

    /// CONDICIÓN D: Hecho Inyectado en Contexto SIN CoT (Control RAG Puro)
    /// Scaffolding: Inyecta el hecho directamente en el sistema o contexto sin ninguna envoltura de pensamiento
    #[test]
    fn test_condition_d_context_without_cot() {
        println!("\n========================================================");
        println!("📦 EVALUACIÓN EMPÍRICA — CONDICIÓN D: CONTEXTO DIRECTO (SIN CoT)");
        println!("Modelo: Qwen2.5-0.5B-Instruct (models/production/qwen2_5_0_5b.gaje) [Q4_0]");
        println!("========================================================\n");

        let test_cases = get_test_cases();
        let path = "models/production/qwen2_5_0_5b.gaje";
        let reader = GajeFlatFileReader::open(path).expect("Open Qwen Q4_0");
        let tokenizer = reader.get_embedded_gtok().expect("Get GTOK");
        let mut model = reader.load_genomic().expect("Load GenomicLLM");
        let stop_tokens = vec![151645, 151643];

        let mut correct_count = 0;
        let total_start = Instant::now();

        for tc in &test_cases {
            let chat_prompt = format!(
                "<|im_start|>system\nKnowledge: {}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                tc.fact, tc.question
            );
            let input_ids: Vec<usize> = tokenizer.encode(&chat_prompt).into_iter().map(|t| t as usize).collect();

            let output_ids = model.generate_native_core(input_ids, 25, 0.0, 1.15, stop_tokens.clone())
                .expect("Generation failed");

            let output_u32: Vec<u32> = output_ids.iter().map(|&t| t as u32).collect();
            let response_text = tokenizer.decode(&output_u32);

            let is_hit = check_answer(&response_text, tc.correct_pattern, tc.distractor_pattern);
            if is_hit {
                correct_count += 1;
            }

            let status_emoji = if is_hit { "✅ ACIERTO" } else { "❌ FALLO  " };
            println!("[{:02}/20] {} [{}] Q: {}", tc.id, status_emoji, tc.domain, tc.question);
            println!("       D Output: \"{}\"", response_text.trim().replace('\n', " "));
        }

        let elapsed = total_start.elapsed();
        let accuracy_pct = (correct_count as f32 / test_cases.len() as f32) * 100.0;

        println!("\n========================================================");
        println!("📊 RESUMEN FINAL — CONDICIÓN D (CONTEXTO DIRECTO SIN CoT)");
        println!("========================================================");
        println!("Aciertos Válidos:            {} / {}", correct_count, test_cases.len());
        println!("Exactitud Condición D:       {:.1}%", accuracy_pct);
        println!("Tiempo Total:                {:.2?}", elapsed);
        println!("========================================================\n");
    }

    /// CONDICIÓN C5: Canal System con Formato CoT [THINKING][QUERY]->fact [ANSWER]
    /// Aísla si el formato estructurado ayuda cuando se inyecta en el system prompt (vs D que es texto plano)
    #[test]
    fn test_condition_c5_system_cot() {
        println!("\n========================================================");
        println!("🏛️ EVALUACIÓN EMPÍRICA — CONDICIÓN C5: CANAL SYSTEM CON FORMATO CoT");
        println!("Modelo: Qwen2.5-0.5B-Instruct (models/production/qwen2_5_0_5b.gaje) [Q4_0]");
        println!("========================================================\n");

        let test_cases = get_test_cases();
        let path = "models/production/qwen2_5_0_5b.gaje";
        let reader = GajeFlatFileReader::open(path).expect("Open Qwen Q4_0");
        let tokenizer = reader.get_embedded_gtok().expect("Get GTOK");
        let mut model = reader.load_genomic().expect("Load GenomicLLM");
        let stop_tokens = vec![151645, 151643];

        let mut correct_count = 0;
        let total_start = Instant::now();

        for tc in &test_cases {
            let chat_prompt = format!(
                "<|im_start|>system\n[THINKING] [QUERY: {}] -> {}\n[ANSWER]<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                tc.query_term, tc.fact, tc.question
            );
            let input_ids: Vec<usize> = tokenizer.encode(&chat_prompt).into_iter().map(|t| t as usize).collect();

            let output_ids = model.generate_native_core(input_ids, 25, 0.0, 1.15, stop_tokens.clone())
                .expect("Generation failed");

            let output_u32: Vec<u32> = output_ids.iter().map(|&t| t as u32).collect();
            let response_text = tokenizer.decode(&output_u32);

            let is_hit = check_answer(&response_text, tc.correct_pattern, tc.distractor_pattern);
            if is_hit {
                correct_count += 1;
            }

            let status_emoji = if is_hit { "✅ ACIERTO" } else { "❌ FALLO  " };
            println!("[{:02}/20] {} [{}] Q: {}", tc.id, status_emoji, tc.domain, tc.question);
            println!("       C5 Output: \"{}\"", response_text.trim().replace('\n', " "));
        }

        let elapsed = total_start.elapsed();
        let accuracy_pct = (correct_count as f32 / test_cases.len() as f32) * 100.0;

        println!("\n========================================================");
        println!("📊 RESUMEN FINAL — CONDICIÓN C5 (SYSTEM + FORMATO CoT)");
        println!("========================================================");
        println!("Aciertos Válidos:            {} / {}", correct_count, test_cases.len());
        println!("Exactitud Condición C5:      {:.1}%", accuracy_pct);
        println!("Tiempo Total:                {:.2?}", elapsed);
        println!("========================================================\n");
    }

    /// CONDICIÓN C4: Canal Assistant Plano ({fact}\nAnswer: )
    /// Aísla si el canal assistant es el único responsable del salto (eliminando todo andamiaje CoT)
    #[test]
    fn test_condition_c4_assistant_plain() {
        println!("\n========================================================");
        println!("⚡ EVALUACIÓN EMPÍRICA — CONDICIÓN C4: CANAL ASSISTANT PLANO");
        println!("Modelo: Qwen2.5-0.5B-Instruct (models/production/qwen2_5_0_5b.gaje) [Q4_0]");
        println!("========================================================\n");

        let test_cases = get_test_cases();
        let path = "models/production/qwen2_5_0_5b.gaje";
        let reader = GajeFlatFileReader::open(path).expect("Open Qwen Q4_0");
        let tokenizer = reader.get_embedded_gtok().expect("Get GTOK");
        let mut model = reader.load_genomic().expect("Load GenomicLLM");
        let stop_tokens = vec![151645, 151643];

        let mut correct_count = 0;
        let total_start = Instant::now();

        for tc in &test_cases {
            let chat_prompt = format!(
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}\nAnswer: ",
                tc.question, tc.fact
            );
            let input_ids: Vec<usize> = tokenizer.encode(&chat_prompt).into_iter().map(|t| t as usize).collect();

            let output_ids = model.generate_native_core(input_ids, 25, 0.0, 1.15, stop_tokens.clone())
                .expect("Generation failed");

            let output_u32: Vec<u32> = output_ids.iter().map(|&t| t as u32).collect();
            let response_text = tokenizer.decode(&output_u32);

            let is_hit = check_answer(&response_text, tc.correct_pattern, tc.distractor_pattern);
            if is_hit {
                correct_count += 1;
            }

            let status_emoji = if is_hit { "✅ ACIERTO" } else { "❌ FALLO  " };
            println!("[{:02}/20] {} [{}] Q: {}", tc.id, status_emoji, tc.domain, tc.question);
            println!("       C4 Output: \"{}\"", response_text.trim().replace('\n', " "));
        }

        let elapsed = total_start.elapsed();
        let accuracy_pct = (correct_count as f32 / test_cases.len() as f32) * 100.0;

        println!("\n========================================================");
        println!("📊 RESUMEN FINAL — CONDICIÓN C4 (ASSISTANT PLANO)");
        println!("========================================================");
        println!("Aciertos Válidos:            {} / {}", correct_count, test_cases.len());
        println!("Exactitud Condición C4:      {:.1}%", accuracy_pct);
        println!("Tiempo Total:                {:.2?}", elapsed);
        println!("========================================================\n");
    }

    /// CONDICIÓN C3: Canal Assistant Estructurado Sin Tags (Fact: {fact}\nAnswer: )
    /// Aísla si los tags especiales aportan algo vs formato estructurado genérico
    #[test]
    fn test_condition_c3_assistant_structure_no_tags() {
        println!("\n========================================================");
        println!("🏷️ EVALUACIÓN EMPÍRICA — CONDICIÓN C3: ESTRUCTURA SIN TAGS");
        println!("Modelo: Qwen2.5-0.5B-Instruct (models/production/qwen2_5_0_5b.gaje) [Q4_0]");
        println!("========================================================\n");

        let test_cases = get_test_cases();
        let path = "models/production/qwen2_5_0_5b.gaje";
        let reader = GajeFlatFileReader::open(path).expect("Open Qwen Q4_0");
        let tokenizer = reader.get_embedded_gtok().expect("Get GTOK");
        let mut model = reader.load_genomic().expect("Load GenomicLLM");
        let stop_tokens = vec![151645, 151643];

        let mut correct_count = 0;
        let total_start = Instant::now();

        for tc in &test_cases {
            let chat_prompt = format!(
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\nFact: {}\nAnswer: ",
                tc.question, tc.fact
            );
            let input_ids: Vec<usize> = tokenizer.encode(&chat_prompt).into_iter().map(|t| t as usize).collect();

            let output_ids = model.generate_native_core(input_ids, 25, 0.0, 1.15, stop_tokens.clone())
                .expect("Generation failed");

            let output_u32: Vec<u32> = output_ids.iter().map(|&t| t as u32).collect();
            let response_text = tokenizer.decode(&output_u32);

            let is_hit = check_answer(&response_text, tc.correct_pattern, tc.distractor_pattern);
            if is_hit {
                correct_count += 1;
            }

            let status_emoji = if is_hit { "✅ ACIERTO" } else { "❌ FALLO  " };
            println!("[{:02}/20] {} [{}] Q: {}", tc.id, status_emoji, tc.domain, tc.question);
            println!("       C3 Output: \"{}\"", response_text.trim().replace('\n', " "));
        }

        let elapsed = total_start.elapsed();
        let accuracy_pct = (correct_count as f32 / test_cases.len() as f32) * 100.0;

        println!("\n========================================================");
        println!("📊 RESUMEN FINAL — CONDICIÓN C3 (ESTRUCTURA SIN TAGS)");
        println!("========================================================");
        println!("Aciertos Válidos:            {} / {}", correct_count, test_cases.len());
        println!("Exactitud Condición C3:      {:.1}%", accuracy_pct);
        println!("Tiempo Total:                {:.2?}", elapsed);
        println!("========================================================\n");
    }

    /// CONDICIÓN C2: Canal Assistant con Tags SIN [THINKING] ([QUERY: X] -> fact\n[ANSWER] )
    /// Aísla si la palabra/etiqueta [THINKING] aporta algo sobre el enlace asociativo directo
    #[test]
    fn test_condition_c2_assistant_no_thinking() {
        println!("\n========================================================");
        println!("🔍 EVALUACIÓN EMPÍRICA — CONDICIÓN C2: TAGS SIN [THINKING]");
        println!("Modelo: Qwen2.5-0.5B-Instruct (models/production/qwen2_5_0_5b.gaje) [Q4_0]");
        println!("========================================================\n");

        let test_cases = get_test_cases();
        let path = "models/production/qwen2_5_0_5b.gaje";
        let reader = GajeFlatFileReader::open(path).expect("Open Qwen Q4_0");
        let tokenizer = reader.get_embedded_gtok().expect("Get GTOK");
        let mut model = reader.load_genomic().expect("Load GenomicLLM");
        let stop_tokens = vec![151645, 151643];

        let mut correct_count = 0;
        let total_start = Instant::now();

        for tc in &test_cases {
            let chat_prompt = format!(
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n[QUERY: {}] -> {}\n[ANSWER] ",
                tc.question, tc.query_term, tc.fact
            );
            let input_ids: Vec<usize> = tokenizer.encode(&chat_prompt).into_iter().map(|t| t as usize).collect();

            let output_ids = model.generate_native_core(input_ids, 25, 0.0, 1.15, stop_tokens.clone())
                .expect("Generation failed");

            let output_u32: Vec<u32> = output_ids.iter().map(|&t| t as u32).collect();
            let response_text = tokenizer.decode(&output_u32);

            let is_hit = check_answer(&response_text, tc.correct_pattern, tc.distractor_pattern);
            if is_hit {
                correct_count += 1;
            }

            let status_emoji = if is_hit { "✅ ACIERTO" } else { "❌ FALLO  " };
            println!("[{:02}/20] {} [{}] Q: {}", tc.id, status_emoji, tc.domain, tc.question);
            println!("       C2 Output: \"{}\"", response_text.trim().replace('\n', " "));
        }

        let elapsed = total_start.elapsed();
        let accuracy_pct = (correct_count as f32 / test_cases.len() as f32) * 100.0;

        println!("\n========================================================");
        println!("📊 RESUMEN FINAL — CONDICIÓN C2 (TAGS SIN THINKING)");
        println!("========================================================");
        println!("Aciertos Válidos:            {} / {}", correct_count, test_cases.len());
        println!("Exactitud Condición C2:      {:.1}%", accuracy_pct);
        println!("Tiempo Total:                {:.2?}", elapsed);
        println!("========================================================\n");
    }

    fn embed_text_gtok(
        text: &str,
        model: &GenomicLLM,
        tokenizer: &GtokNativeTokenizer,
    ) -> Vec<f32> {
        let tokens_u32 = tokenizer.encode(text);
        let stop_tokens = tokenizer.get_stop_tokens();
        let dim = model.dim();
        let mut sum_vec = vec![0.0f32; dim];
        let mut total_w = 0.0f32;

        let is_stopword = |w: &str| -> bool {
            matches!(
                w,
                "de" | "la" | "el" | "en" | "es" | "y" | "a" | "un" | "una" | "unos" | "unas"
                    | "con" | "por" | "para" | "que" | "del" | "los" | "las" | "the" | "is"
                    | "of" | "and" | "in" | "to" | "," | "." | ";" | ":" | "¿" | "?" | "!" | "¡"
            )
        };

        for &tok_u32 in &tokens_u32 {
            let tok = tok_u32 as usize;
            if stop_tokens.contains(&tok_u32) {
                continue;
            }
            let tok_str = tokenizer.decode(&[tok_u32]).trim().to_lowercase();
            let w = if is_stopword(&tok_str) { 0.15f32 } else { 1.0f32 };

            if let Ok(emb) = model.get_token_embedding(tok) {
                if emb.len() == dim {
                    for i in 0..dim {
                        sum_vec[i] += emb[i] * w;
                    }
                    total_w += w;
                }
            }
        }

        if total_w > 0.0 {
            let inv_w = 1.0 / total_w;
            for v in &mut sum_vec {
                *v *= inv_w;
            }
        }

        let norm = sum_vec.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 1e-9 {
            let inv_norm = 1.0 / norm;
            for v in &mut sum_vec {
                *v *= inv_norm;
            }
        }

        sum_vec
    }

    /// TEST DE RETRIEVAL REAL .GMEM + GENERACIÓN SIN PREFIJO (RAG E2E Auténtico)
    /// 1. Ingesta los 20 hechos en un índice .gmem mediante embed_text_gtok.
    /// 2. Para cada pregunta, busca el vecino más cercano por similitud coseno (Recall@1).
    /// 3. Inyecta el hecho recuperado en el System Prompt.
    /// 4. El asistente responde LIBREMENTE desde <|im_start|>assistant\n (sin prefijo ni completion).
    #[test]
    fn test_e2e_real_gmem_rag_retrieval_and_generation() {
        println!("\n========================================================");
        println!("🧠 EVALUACIÓN EMPÍRICA — RAG E2E REAL (.GMEM RETRIEVAL + SYSTEM)");
        println!("Modelo: Qwen2.5-0.5B-Instruct (models/production/qwen2_5_0_5b.gaje) [Q4_0]");
        println!("========================================================\n");

        let test_cases = get_test_cases();
        let path = "models/production/qwen2_5_0_5b.gaje";
        let reader = GajeFlatFileReader::open(path).expect("Open Qwen Q4_0");
        let tokenizer = reader.get_embedded_gtok().expect("Get GTOK");
        let mut model = reader.load_genomic().expect("Load GenomicLLM");
        let stop_tokens = vec![151645, 151643];

        let dim = model.dim() as u32;
        let mut memory_index = GmemMemoryIndex::new(dim);

        println!("📥 1. Ingestando los 20 hechos en .gmem usando embed_text_gtok ({}d)...", dim);
        let t_embed_start = Instant::now();
        for tc in &test_cases {
            let vec = embed_text_gtok(tc.fact, &model, &tokenizer);
            memory_index.add_entry(tc.id as u64, vec, tc.fact.to_string());
        }
        println!("   ✅ Ingesta completada en {:.2?}\n", t_embed_start.elapsed());

        let mut retrieval_hits = 0;
        let mut generation_hits = 0;
        let total_start = Instant::now();

        println!("🔍 2. Ejecutando Retrieval Vectorial y Generación Libre sin Prefijo...");
        for tc in &test_cases {
            // A. Retrieval
            let q_vec = embed_text_gtok(tc.question, &model, &tokenizer);
            let matches = memory_index.search_top_k(&q_vec, 1);
            assert!(!matches.is_empty(), "Matches should not be empty");

            let retrieved_entry = matches[0].0;
            let sim = matches[0].1;
            let is_retrieval_correct = retrieved_entry.id == tc.id as u64;
            if is_retrieval_correct {
                retrieval_hits += 1;
            }

            // B. Generación Libre RAG (sin prefijo en el turno del asistente)
            let chat_prompt = format!(
                "<|im_start|>system\nKnowledge: {}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                retrieved_entry.text, tc.question
            );
            let input_ids: Vec<usize> = tokenizer.encode(&chat_prompt).into_iter().map(|t| t as usize).collect();

            let output_ids = model.generate_native_core(input_ids, 25, 0.0, 1.15, stop_tokens.clone())
                .expect("Generation failed");

            let output_u32: Vec<u32> = output_ids.iter().map(|&t| t as u32).collect();
            let response_text = tokenizer.decode(&output_u32);

            let is_gen_hit = check_answer(&response_text, tc.correct_pattern, tc.distractor_pattern);
            if is_gen_hit {
                generation_hits += 1;
            }

            let ret_emoji = if is_retrieval_correct { "🎯 MATCH" } else { "⚠️ MISS " };
            let gen_emoji = if is_gen_hit { "✅ HIT " } else { "❌ MISS" };

            println!(
                "[{:02}/20] Ret: {} (sim: {:.3}) | Gen: {} [{}] Q: {}",
                tc.id, ret_emoji, sim, gen_emoji, tc.domain, tc.question
            );
            println!("       Retrieved: \"{}\"", retrieved_entry.text);
            println!("       Output:    \"{}\"", response_text.trim().replace('\n', " "));
        }

        let elapsed = total_start.elapsed();
        let ret_pct = (retrieval_hits as f32 / test_cases.len() as f32) * 100.0;
        let gen_pct = (generation_hits as f32 / test_cases.len() as f32) * 100.0;

        println!("\n========================================================");
        println!("📊 RESUMEN FINAL — RAG E2E REAL (.GMEM + GENERACIÓN LIBRE)");
        println!("========================================================");
        println!("Top-1 Retrieval Recall (.gmem): {} / {} ({:.1}%)", retrieval_hits, test_cases.len(), ret_pct);
        println!("Exactitud Generación E2E (LLM): {} / {} ({:.1}%)", generation_hits, test_cases.len(), gen_pct);
        println!("Tiempo Total:                   {:.2?}", elapsed);
        println!("========================================================\n");
    }
}


