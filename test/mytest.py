from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

# ============================================================
# 1. 路径配置
# ============================================================

BASE_DIR = Path(r"D:\MyWork\Qwen3_06B")

# 你训练好的模型权重目录
MODEL_DIR = BASE_DIR / "output" / "final"

# 你自己训练好的 tokenizer 目录
TOKENIZER_DIR = BASE_DIR / "make_tokenizer"/ "tokenizer"


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


# ============================================================
# 4. 检查 tokenizer 和模型词表大小
# ============================================================

assert len(tokenizer) == model.config.vocab_size, (
    f"Tokenizer 和模型词表大小不一致: "
    f"{len(tokenizer)} != {model.config.vocab_size}"
)


# ============================================================
# 5. 准备输入
# ============================================================

prompt = "请用中文介绍一下大型语言模型。"

messages = [
    {
        "role": "user",
        "content": prompt,
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)

model_inputs = tokenizer(
    [text],
    return_tensors="pt",
).to(model.device)


# ============================================================
# 6. 生成
# ============================================================

with torch.no_grad():
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )


# ============================================================
# 7. 解码
# ============================================================

output_ids = generated_ids[
    0
][len(model_inputs.input_ids[0]):].tolist()

content = tokenizer.decode(
    output_ids,
    skip_special_tokens=True,
)

print("\n" + "=" * 60)
print("PROMPT:")
print(prompt)

print("\n" + "=" * 60)
print("OUTPUT:")
print(content)