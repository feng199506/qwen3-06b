from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

# ============================================================
# 1. 路径配置
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "lm_output" / "final"
TOKENIZER_DIR = BASE_DIR / "tokenizer"


# ============================================================
# 2. 加载 tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    str(TOKENIZER_DIR),
    trust_remote_code=True,
)

print("Tokenizer loaded")
print("Tokenizer vocab_size:", tokenizer.vocab_size)
print("Tokenizer length:", len(tokenizer))


# ============================================================
# 3. 加载自己训练的模型
# ============================================================

model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_DIR),
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)

model.eval()

print("Model loaded")
print("Model device:", model.device)
print("Model vocab_size:", model.config.vocab_size)

assert len(tokenizer) == model.config.vocab_size, (
    f"Tokenizer/model vocab mismatch: "
    f"{len(tokenizer)} != {model.config.vocab_size}"
)


# ============================================================
# 4. 生成测试
# ============================================================

prompt = "人工智能是一种"

inputs = tokenizer(
    prompt,
    return_tensors="pt",
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=False))
