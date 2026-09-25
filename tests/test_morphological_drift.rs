use _impl::nn::repl::load_model_and_tokenizer;
use std::path::Path;

#[test]
fn test_morphological_drift_isolation() {
    let p = "models/production/qwen2_5_1_5b.gaje";
    if !Path::new(p).exists() {
        println!("Modelo no encontrado: {}", p);
        return;
    }

    println!("\n=======================================================");
    println!("🔬 EXPERIMENTO DE AISLAMIENTO DE DERIVA MORFOLÓGICA");
    println!("=======================================================\n");

    let (mut llm, tokenizer) = load_model_and_tokenizer(p).expect("cargar modelo");
    let eos_ids: Vec<usize> = tokenizer
        .get_stop_tokens()
        .into_iter()
        .map(|t| t as usize)
        .collect();

    let user_msg = "¿Cuál es la capital de Argentina?";

    struct TestCase {
        name: &'static str,
        system_prompt: Option<&'static str>,
        repetition_penalty: f32,
    }

    let cases = vec![
        TestCase {
            name: "Caso 1: Bare Prompt (sin system, N=10 tokens, rep=1.0)",
            system_prompt: None,
            repetition_penalty: 1.0,
        },
        TestCase {
            name: "Caso 2: Bare Prompt con rep_penalty=1.15",
            system_prompt: None,
            repetition_penalty: 1.15,
        },
        TestCase {
            name: "Caso 3: System Prompt corto neutro (rep=1.0)",
            system_prompt: Some("Eres un asistente útil y conciso."),
            repetition_penalty: 1.0,
        },
        TestCase {
            name: "Caso 4: Contexto de memoria inyectado ('Buenos Aires es la capital...') con rep=1.0",
            system_prompt: Some("Eres un asistente útil.\n\nInformación de contexto:\n- Buenos Aires es la capital de Argentina."),
            repetition_penalty: 1.0,
        },
        TestCase {
            name: "Caso 5: Contexto de memoria inyectado ('Buenos Aires es la capital...') con rep=1.15",
            system_prompt: Some("Eres un asistente útil.\n\nInformación de contexto:\n- Buenos Aires es la capital de Argentina."),
            repetition_penalty: 1.15,
        },
        TestCase {
            name: "Caso 6: Long System neutro (sin palabras 'capital'/'es'/'Argentina', rep=1.0)",
            system_prompt: Some("El sistema solar comprende una variedad de planetas rocosos y gaseosos que orbitan alrededor de una estrella central clasificada como enana amarilla de tipo espectral G2V. La física celeste y las leyes de Kepler gobiernan sus trayectorias orbitales con precisión matemática comprobable mediante observaciones astronómicas continuas."),
            repetition_penalty: 1.0,
        },
        TestCase {
            name: "Caso 7: Long System neutro con rep=1.15",
            system_prompt: Some("El sistema solar comprende una variedad de planetas rocosos y gaseosos que orbitan alrededor de una estrella central clasificada como enana amarilla de tipo espectral G2V. La física celeste y las leyes de Kepler gobiernan sus trayectorias orbitales con precisión matemática comprobable mediante observaciones astronómicas continuas."),
            repetition_penalty: 1.15,
        },
    ];

    for tc in cases {
        let chat_prompt = match tc.system_prompt {
            Some(sys) => format!(
                "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                sys, user_msg
            ),
            None => format!(
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                user_msg
            ),
        };

        let prompt_tokens: Vec<usize> = tokenizer
            .encode(&chat_prompt, false)
            .unwrap()
            .into_iter()
            .map(|t| t as usize)
            .collect();

        llm.clear_cache_core();
        let gen_tokens = llm
            .generate_native_core(
                prompt_tokens.clone(),
                40,
                0.0,
                tc.repetition_penalty,
                eos_ids.clone(),
            )
            .expect("generación");

        let gen_u32: Vec<u32> = gen_tokens.into_iter().map(|t| t as u32).collect();
        let raw_reply = tokenizer.decode(&gen_u32, true).unwrap_or_default();
        let clean = raw_reply.split("<|im_end|>").next().unwrap_or(&raw_reply).trim();

        println!("-------------------------------------------------------");
        println!("📋 {}", tc.name);
        println!("   • Prompt tokens: {}", prompt_tokens.len());
        println!("   • Repetition Penalty: {}", tc.repetition_penalty);
        println!("   • Respuesta generada: {:?}", clean);
        println!("   • Contiene 'capitale': {}", clean.contains("capitale"));
        println!("   • Contiene 'capital':  {}", clean.contains("capital"));
        println!("   • Contiene 'e s':      {}", clean.contains("e s"));
        println!("   • Contiene ' es ':     {}", clean.contains(" es "));
    }
    println!("-------------------------------------------------------\n");
}
