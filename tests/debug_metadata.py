import gguf
import os

model_path = "./data/models/qwen2-0_5b-instruct-fp16.gguf"
reader = gguf.GGUFReader(model_path)

print(f"Model: {os.path.basename(model_path)}")
for field_name in reader.fields:
    if (
        "rope" in field_name
        or "head_count" in field_name
        or "embedding_length" in field_name
    ):
        field = reader.fields[field_name]
        val = field.parts[-1][0]
        print(f"{field_name}: {val} (Type: {type(val)})")
