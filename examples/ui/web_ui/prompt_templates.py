"""GAJE-Flow Prompt Templates Module.

Manages architecture-specific prompt formatting for chat models (Qwen2, SmolLM2, LLaMA, Gemma).
"""


def format_prompt(model_name: str, message: str) -> str:
    """Format an incoming user message into the correct template for the given model architecture."""
    model_name_lower = model_name.lower()

    if "smollm" in model_name_lower or "qwen" in model_name_lower:
        # ChatML template
        return (
            f"<|im_start|>system\nYou are a helpful and precise assistant.<|im_end|>\n"
            f"<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"
        )
    elif "gemma" in model_name_lower:
        # Gemma 2B instruction template
        return f"<start_of_turn>user\n{message}<end_of_turn>\n<start_of_turn>model\n"
    else:
        # Fallback standard prompt
        return f"User: {message}\nAssistant:"


def get_stop_tokens(model_name: str, tokenizer) -> list:
    """Get the appropriate EOS / stop token IDs for the model."""
    eos_ids = [2, 151643, 151645]
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)
    return eos_ids
