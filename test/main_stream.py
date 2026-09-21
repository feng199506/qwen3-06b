from pathlib import Path
from threading import Thread

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)


# ============================================================
# 1. 本地模型路径
# ============================================================

MODEL_DIR = (
    Path(__file__).resolve().parent.parent / "configs"
)


# ============================================================
# 2. 加载 tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_DIR),
    local_files_only=True,
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
    local_files_only=True,
)

model.eval()

print("Model loaded")
print("Model device:", model.device)
print("Model vocab_size:", model.config.vocab_size)


# ============================================================
# 4. 检查 tokenizer 和模型词表
# ============================================================

# assert len(tokenizer) == model.config.vocab_size, (
#     f"Tokenizer 和模型词表大小不一致: "
#     f"{len(tokenizer)} != {model.config.vocab_size}"
# )


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
# 6. 创建流式输出器
# ============================================================

streamer = TextIteratorStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=True,
)


# ============================================================
# 7. 生成参数
# ============================================================

generation_kwargs = {
    **model_inputs,
    "streamer": streamer,
    "max_new_tokens": 512,
    "do_sample": False,
    "pad_token_id": tokenizer.eos_token_id,
}


# ============================================================
# 8. 启动生成线程
# ============================================================

thread = Thread(
    target=model.generate,
    kwargs=generation_kwargs,
)

thread.start()


# ============================================================
# 9. ChatGPT 风格流式输出
# ============================================================

print("\n" + "=" * 60)
print("PROMPT:")
print(prompt)

print("\n" + "=" * 60)
print("OUTPUT:")
print("=" * 60)

for new_text in streamer:

    print(
        new_text,
        end="",
        flush=True,
    )


# ============================================================
# 10. 等待生成结束
# ============================================================

thread.join()

print("\n\n" + "=" * 60)
print("DONE")