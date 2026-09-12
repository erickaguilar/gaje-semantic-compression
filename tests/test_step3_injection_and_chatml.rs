use _impl::compute::island::{
    configure_model_memory, detect_chat_template_from_tokenizer,
    format_chat_prompt_from_template, prepare_prompt_with_memory,
    IslandNiche, IslandOrchestrator, IslandSearchResult,
    MemoryConfig, MemoryDecision,
};
use _impl::core::gtok::ChatTemplate;
use _impl::nn::repl::load_model_and_tokenizer;
use std::path::Path;

#[test]
fn test_step3_chatml_and_llama3_pipeline_ordering() {
    let mut orch = IslandOrchestrator::new(4);
    orch.min_similarity = 0.40;
    orch.episodic_min_sim = 0.40;
    orch.documental_min_sim = 0.50;

    let matches = vec![
        IslandSearchResult {
            niche: IslandNiche::Episodic,
            id: 1,
            similarity: 0.88,
            text: "La capital de Francia es París.".to_string(),
        },
        IslandSearchResult {
            niche: IslandNiche::Documental,
            id: 2,
            similarity: 0.65, // Delta_top = 0.88 - 0.65 = 0.23 >= 0.12 (Entropy gap pasa); 0.65 < 0.88 * 0.90 (0.792) -> Podado por K-WTA
            text: "Francia es un país soberano de Europa occidental.".to_string(),
        },
    ];

    let original_sys = "Tu nombre es GAJE. Eres un asistente conciso y preciso.";
    let user_msg = "¿Cuál es la capital de Francia?";

    // 1. Verificación del bloque de sistema aumentado
    let (augmented_sys, decision, facts) =
        orch.build_augmented_system_prompt(original_sys, &matches, 128);

    assert!(matches!(decision, MemoryDecision::Injected { .. }));
    assert_eq!(facts.len(), 1, "Debe inyectarse el recuerdo dominante (el 2do podado por K-WTA)");
    assert!(augmented_sys.contains("Tu nombre es GAJE"));
    assert!(augmented_sys.contains("[Conocimiento Recuperado:"));
    assert!(augmented_sys.contains("La capital de Francia es París."));
    assert!(!augmented_sys.contains("Francia es un país soberano"), "K-WTA debe podar el recuerdo dominado");

    // 2. Verificación de orden estricto en ChatML:
    // El conocimiento DEBE quedar DENTRO de <|im_start|>system ... <|im_end|>
    // y ANTES de <|im_start|>user y <|im_start|>assistant.
    let chatml_prompt = format_chat_prompt_from_template(
        ChatTemplate::ChatML,
        &augmented_sys,
        user_msg,
        None,
    );

    let sys_start = chatml_prompt.find("<|im_start|>system").expect("system start");
    let knowledge_pos = chatml_prompt.find("[Conocimiento Recuperado:").expect("knowledge pos");
    let sys_end = chatml_prompt.find("<|im_end|>").expect("system end");
    let user_start = chatml_prompt.find("<|im_start|>user").expect("user start");
    let asst_start = chatml_prompt.find("<|im_start|>assistant").expect("asst start");

    assert!(sys_start < knowledge_pos, "System tag must precede knowledge");
    assert!(knowledge_pos < sys_end, "Knowledge must be inside system turn (before <|im_end|>)");
    assert!(sys_end < user_start, "System turn must close before user turn");
    assert!(user_start < asst_start, "User turn must precede assistant generation prompt");
    assert!(chatml_prompt.ends_with("<|im_start|>assistant\n"), "Must terminate cleanly with assistant generation prompt");

    // 3. Verificación de orden estricto en Llama3:
    // El conocimiento DEBE quedar dentro del bloque <|start_header_id|>system<|end_header_id|> ... <|eot_id|>
    let llama3_prompt = format_chat_prompt_from_template(
        ChatTemplate::Llama3,
        &augmented_sys,
        user_msg,
        None,
    );

    let l3_sys_start = llama3_prompt.find("<|start_header_id|>system<|end_header_id|>").expect("l3 sys");
    let l3_knowledge = llama3_prompt.find("[Conocimiento Recuperado:").expect("l3 knowledge");
    let l3_eot = llama3_prompt.find("<|eot_id|>").expect("l3 eot");
    let l3_user = llama3_prompt.find("<|start_header_id|>user<|end_header_id|>").expect("l3 user");
    let l3_asst = llama3_prompt.find("<|start_header_id|>assistant<|end_header_id|>").expect("l3 asst");

    assert!(l3_sys_start < l3_knowledge);
    assert!(l3_knowledge < l3_eot);
    assert!(l3_eot < l3_user);
    assert!(l3_user < l3_asst);
}

#[test]
fn test_step3_telemetry_all_six_states() {
    let model_path = "models/born/max.gaje";
    if !Path::new(model_path).exists() {
        println!("⚠️ Saltando test con modelo real (archivo no disponible)");
        return;
    }

    let (llm, tokenizer) = load_model_and_tokenizer(model_path).expect("cargar max.gaje");
    let template = detect_chat_template_from_tokenizer(&tokenizer);

    // Estado 1: Disabled (use_memory = false)
    {
        let config = MemoryConfig {
            use_memory: false,
            threshold: 0.42,
            uses_whitening: false,
            mu_vector: None,
            whitening_missing: false,
            max_tokens_context: 128,
        };
        let mut orch = IslandOrchestrator::new(llm.dim() as u32);
        orch.add_memory(IslandNiche::Episodic, 1, vec![0.1; llm.dim()], "Dato".to_string());

        let (prompt, tel) = prepare_prompt_with_memory(
            "¿Pregunta?",
            None,
            "Sys",
            template,
            &llm,
            &tokenizer,
            Some(&orch),
            &config,
        );
        assert_eq!(tel.state, "memory_disabled");
        assert_eq!(tel.facts_injected, 0);
        assert!(!prompt.contains("[Conocimiento"));
    }

    // Estado 2: Empty (memoria sin ningún recuerdo cargado)
    {
        let config = MemoryConfig {
            use_memory: true,
            threshold: 0.42,
            uses_whitening: false,
            mu_vector: None,
            whitening_missing: false,
            max_tokens_context: 128,
        };
        let orch = IslandOrchestrator::new(llm.dim() as u32); // vacío

        let (prompt, tel) = prepare_prompt_with_memory(
            "¿Pregunta?",
            None,
            "Sys",
            template,
            &llm,
            &tokenizer,
            Some(&orch),
            &config,
        );
        assert_eq!(tel.state, "memory_empty");
        assert_eq!(tel.facts_injected, 0);
        assert!(!prompt.contains("[Conocimiento"));
    }

    // Estado 3: DimMismatch (dimensión del modelo != dimensión del índice de memoria)
    {
        let config = MemoryConfig {
            use_memory: true,
            threshold: 0.42,
            uses_whitening: false,
            mu_vector: None,
            whitening_missing: false,
            max_tokens_context: 128,
        };
        let mut orch = IslandOrchestrator::new(128);
        orch.add_memory(IslandNiche::Episodic, 1, vec![0.1; 128], "Dato".to_string());

        let (prompt, tel) = prepare_prompt_with_memory(
            "¿Pregunta?",
            None,
            "Sys",
            template,
            &llm,
            &tokenizer,
            Some(&orch),
            &config,
        );
        assert_eq!(tel.state, "memory_dim_mismatch");
        if let MemoryDecision::DimMismatch { expected, found } = tel.decision {
            assert_eq!(expected, llm.dim());
            assert_eq!(found, 128);
        } else {
            panic!("Expected DimMismatch");
        }
        assert!(!prompt.contains("[Conocimiento"));
    }

    // Estado 4: Injected (consulta relevante supera tau* y entropy gap)
    {
        let text_fact = "La capital de Francia es París.";
        let v_fact = llm.embed_text(text_fact, &tokenizer).expect("embed fact");

        let mut orch = IslandOrchestrator::new(llm.dim() as u32);
        orch.min_similarity = 0.42;
        orch.episodic_min_sim = 0.42;
        orch.documental_min_sim = 0.50;
        orch.add_memory(IslandNiche::Episodic, 1, v_fact, text_fact.to_string());

        let config = MemoryConfig {
            use_memory: true,
            threshold: 0.42,
            uses_whitening: false,
            mu_vector: None,
            whitening_missing: false,
            max_tokens_context: 128,
        };

        let (prompt, tel) = prepare_prompt_with_memory(
            "¿Cuál es la capital de Francia?",
            None,
            "Eres GAJE.",
            template,
            &llm,
            &tokenizer,
            Some(&orch),
            &config,
        );

        assert_eq!(tel.state, "memory_injected");
        assert_eq!(tel.facts_injected, 1);
        assert!(prompt.contains("[Conocimiento Recuperado:"));
        assert!(prompt.contains("La capital de Francia es París."));
    }

    // Estado 5: RejectedLowSimilarity (consulta no relacionada cae por debajo del umbral)
    {
        let text_fact = "La fotosíntesis convierte la luz solar en energía química.";
        let v_fact = llm.embed_text(text_fact, &tokenizer).expect("embed fact");

        let mut orch = IslandOrchestrator::new(llm.dim() as u32);
        orch.min_similarity = 0.88;
        orch.episodic_min_sim = 0.88;
        orch.add_memory(IslandNiche::Episodic, 1, v_fact, text_fact.to_string());

        let config = MemoryConfig {
            use_memory: true,
            threshold: 0.88,
            uses_whitening: false,
            mu_vector: None,
            whitening_missing: false,
            max_tokens_context: 128,
        };

        let (prompt, tel) = prepare_prompt_with_memory(
            "¿Cómo se prepara una pizza margarita?",
            None,
            "Eres GAJE.",
            template,
            &llm,
            &tokenizer,
            Some(&orch),
            &config,
        );

        assert_eq!(tel.state, "rejected_low_similarity");
        assert_eq!(tel.facts_injected, 0);
        assert!(!prompt.contains("[Conocimiento"));
    }

    // Estado 6: RejectedEntropyGap (dos hechos compiten muy cerca con Delta_top < 0.12)
    {
        let v_query = llm.embed_text("¿Qué causó la Primera Guerra Mundial?", &tokenizer).expect("embed q");

        let mut orch = IslandOrchestrator::new(llm.dim() as u32);
        orch.min_similarity = 0.40;
        orch.entropy_gap_threshold = 0.12;

        let mut v1 = v_query.clone();
        v1[0] += 0.01;
        let mut v2 = v_query.clone();
        v2[1] += 0.01;

        orch.add_memory(IslandNiche::Documental, 1, v1, "Tratado de Versalles en 1919".to_string());
        orch.add_memory(IslandNiche::Documental, 2, v2, "Armisticio de Compiègne en 1918".to_string());

        let config = MemoryConfig {
            use_memory: true,
            threshold: 0.40,
            uses_whitening: false,
            mu_vector: None,
            whitening_missing: false,
            max_tokens_context: 128,
        };

        let (prompt, tel) = prepare_prompt_with_memory(
            "¿Qué causó la Primera Guerra Mundial?",
            None,
            "Eres GAJE.",
            template,
            &llm,
            &tokenizer,
            Some(&orch),
            &config,
        );

        assert_eq!(tel.state, "rejected_entropy_gap");
        assert_eq!(tel.facts_injected, 0);
        assert!(!prompt.contains("[Conocimiento"));
    }
}

#[test]
fn test_step3_option_c_mu_fallback() {
    let fake_path = Path::new("models/production/nonexistent_model_896d.gaje");
    let (threshold, uses_whitening, mu, whitening_missing) =
        configure_model_memory(fake_path, 896);

    assert_eq!(threshold, 0.50, "Opción C debe usar tau* sin whitening (0.50)");
    assert!(!uses_whitening, "Whitening debe estar desactivado ante ausencia de .mu.bin");
    assert!(mu.is_none(), "Vector mu debe ser None");
    assert!(whitening_missing, "Flag whitening_missing debe ser true");

    let qwen_path = Path::new("models/production/qwen2_5_0_5b_q2_0.gaje");
    if qwen_path.exists() {
        let (q_tau, q_white, q_mu, q_miss) = configure_model_memory(qwen_path, 896);
        assert_eq!(q_tau, 0.33, "Qwen con .mu.bin debe tener tau*=0.33");
        assert!(q_white, "Qwen con .mu.bin debe tener whitening activo");
        assert!(q_mu.is_some(), "Vector mu debe estar presente");
        assert_eq!(q_mu.unwrap().len(), 896, "Dimensión de mu debe coincidir con 896");
        assert!(!q_miss, "whitening_missing debe ser false");
    }

    let pico_path = Path::new("models/production/gaje_pico_135m.gaje");
    if pico_path.exists() {
        let (p_tau, p_white, p_mu, p_miss) = configure_model_memory(pico_path, 576);
        assert_eq!(p_tau, 0.27, "Pico con .mu.bin debe tener tau*=0.27");
        assert!(p_white, "Pico con .mu.bin debe tener whitening activo");
        assert!(p_mu.is_some(), "Vector mu debe estar presente");
        assert_eq!(p_mu.unwrap().len(), 576, "Dimensión de mu debe coincidir con 576");
        assert!(!p_miss, "whitening_missing debe ser false");
    }
}
