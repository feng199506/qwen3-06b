from pathlib import Path
from threading import Thread

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
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
# 3. 加载模型
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
# 4. 流式生成
# ============================================================

prompt = "人工智能是一种"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

streamer = TextIteratorStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=True,
)

generation_kwargs = dict(
    **inputs,
    max_new_tokens=128,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
    streamer=streamer,
    pad_token_id=tokenizer.eos_token_id,
)

thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

print("OUTPUT:")
for text in streamer:
    print(text, end="", flush=True)
print()
thread.join()
