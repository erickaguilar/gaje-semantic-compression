#[cfg(test)]
mod tests {
    use _impl::io::flat_reader::GajeFlatFileReader;

    #[test]
    fn test_tokenizer_chatml_probe() {
        let path4 = "models/production/qwen2_5_0_5b.gaje";
        let reader4 = GajeFlatFileReader::open(path4).expect("Open Q4");
        let tokenizer = reader4.get_embedded_gtok().expect("Embedded GTOK");

        let prompt_raw = "La capital de Francia es";
        let prompt_chatml = "<|im_start|>user\nLa capital de Francia es<|im_end|>\n<|im_start|>assistant\n";

        let tokens_raw = tokenizer.encode(prompt_raw);
        let tokens_chatml = tokenizer.encode(prompt_chatml);

        println!("\n=== TOKENIZER VERIFICATION PROBE ===");
        println!("Prompt Raw: '{}'", prompt_raw);
        println!("Token IDs Raw: {:?}", tokens_raw);
        for &t in &tokens_raw {
            println!("  ID {}: '{}'", t, tokenizer.decode(&[t]));
        }

        println!("\nPrompt ChatML: '{}'", prompt_chatml);
        println!("Token IDs ChatML: {:?}", tokens_chatml);
        for &t in &tokens_chatml {
            println!("  ID {}: '{}'", t, tokenizer.decode(&[t]));
        }

        // Check special tokens
        let im_start_id = tokenizer.encode("<|im_start|>");
        let im_end_id = tokenizer.encode("<|im_end|>");
        println!("\nSpecial token '<|im_start|>': {:?}", im_start_id);
        println!("Special token '<|im_end|>': {:?}", im_end_id);
        let mut m_ctrl = reader4.load_genomic().expect("Load Q4");

        // Caso 3: Inglés canónico
        let mut en_tokens: Vec<usize> = vec![151644, 872, 198];
        en_tokens.extend(tokenizer.encode("What is the capital of France?").into_iter().map(|t| t as usize));
        en_tokens.extend(vec![151645, 198, 151644, 77091, 198]);

        println!("\n--- GENERACIÓN EN INGLÉS (Q4_0) ---");
        println!("Prompt: 'What is the capital of France?'");
        let gen3 = m_ctrl.generate_native_core(en_tokens, 20, 0.0, 1.1, vec![151645, 151643]).unwrap();
        let gen3_u32: Vec<u32> = gen3.iter().map(|&t| t as u32).collect();
        println!("Output: '{}'", tokenizer.decode(&gen3_u32));
    }
}
