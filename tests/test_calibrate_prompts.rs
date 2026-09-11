#[cfg(test)]
mod tests {
    use _impl::io::flat_reader::GajeFlatFileReader;

    fn build_chatml_tokens(tokenizer: &_impl::core::gtok::GtokNativeTokenizer, user_text: &str) -> Vec<usize> {
        let mut tokens = vec![151644, 872, 198]; // <|im_start|>user\n
        tokens.extend(tokenizer.encode(user_text).into_iter().map(|t| t as usize));
        tokens.extend(vec![151645, 198, 151644, 77091, 198]); // <|im_end|>\n<|im_start|>assistant\n
        tokens
    }

    #[test]
    fn test_calibrate_prompts_q4() {
        let path4 = "models/production/qwen2_5_0_5b.gaje";
        let reader4 = GajeFlatFileReader::open(path4).expect("Open Q4");
        let tokenizer = reader4.get_embedded_gtok().expect("Embedded GTOK");
        let mut m_ctrl = reader4.load_genomic().expect("Load Q4");

        let candidate_prompts = [
            ("EN Factual", "What is the capital of France?"),
            ("EN Science", "What is DNA?"),
            ("EN Completion", "The sun is a star located at the center of the"),
            ("ZH Science", "太阳系中最大的行星是哪一颗？"),
            ("ZH Factual", "法国的首都是哪座城市？"),
            ("ES Directo", "Responde únicamente el nombre de la ciudad: ¿cuál es la capital de Francia?"),
            ("ES Completar", "El agua está compuesta por hidrógeno y"),
            ("ES Definición", "¿Qué es la fotosíntesis?"),
        ];

        println!("\n=== CALIBRACIÓN DE PROMPTS EN CONTROL Q4_0 (Base de Verdad) ===");
        let eos_ids = vec![151645, 151643];

        for (tag, prompt) in &candidate_prompts {
            let tokens = build_chatml_tokens(&tokenizer, prompt);
            let gen = m_ctrl.generate_native_core(tokens, 25, 0.0, 1.1, eos_ids.clone()).unwrap();
            let gen_u32: Vec<u32> = gen.iter().map(|&t| t as u32).collect();
            let text = tokenizer.decode(&gen_u32);
            println!("\n[{}] Prompt: '{}'", tag, prompt);
            println!("  Output Q4: '{}'", text.trim());
        }
    }
}
