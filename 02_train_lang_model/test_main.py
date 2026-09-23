# from pathlib import Path
# from transformers import AutoModelForCausalLM, AutoTokenizer
#
# model_name = "Qwen/Qwen3-0.6B"
# cache_dir = Path("./models/huggingface_cache").resolve()
#
# cache_dir.mkdir(parents=True, exist_ok=True)
#
#
# # load the tokenizer and the model
# tokenizer = AutoTokenizer.from_pretrained(
#     model_name,
#     cache_dir=str(cache_dir)
# )
#
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     cache_dir=str(cache_dir),
#     torch_dtype="auto",
#     device_map="auto"
# )
#
# # prepare the model input
# prompt = "Give me a short introduction to large language model."
# messages = [
#     {"role": "user", "content": prompt}
# ]
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True,
#     enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
# )
# model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
#
# # conduct text completion
# generated_ids = model.generate(
#     **model_inputs,
#     max_new_tokens=32768
# )
# output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
#
# # parsing thinking content
# try:
#     # rindex finding 151668 (</think>)
#     index = len(output_ids) - output_ids[::-1].index(151668)
# except ValueError:
#     index = 0
#
# thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
# content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
#
# print("thinking content:", thinking_content)
# print("content:", content)
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# 1. 本地模型路径
# ============================================================

MODEL_DIR = Path(__file__).resolve().parent.parent / "configs"


# ============================================================
# 2. 加载本地 tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_DIR),
    local_files_only=True,
)


# ============================================================
# 3. 加载本地模型
# ============================================================

model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_DIR),
    torch_dtype="auto",
    device_map="auto",
    local_files_only=True,
)

model.eval()


# ============================================================
# 4. 打印信息
# ============================================================

print("Tokenizer loaded")
print("Tokenizer vocab_size:", tokenizer.vocab_size)
print("Tokenizer length:", len(tokenizer))

print("Model loaded")
print("Model device:", model.device)
print("Model vocab_size:", model.config.vocab_size)


# ============================================================
# 5. 准备模型输入
# ============================================================

prompt = "Give me a short introduction to large language model."

messages = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)

model_inputs = tokenizer(
    [text],
    return_tensors="pt",
).to(model.device)


# ============================================================
# 6. 生成
# ============================================================

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,
)


# ============================================================
# 7. 解码
# ============================================================

output_ids = generated_ids[
    0
][len(model_inputs.input_ids[0]):].tolist()


# ============================================================
# 8. 解析 thinking
# ============================================================

try:
    # </think> token id
    index = len(output_ids) - output_ids[::-1].index(151668)

except ValueError:
    index = 0


thinking_content = tokenizer.decode(
    output_ids[:index],
    skip_special_tokens=True,
).strip("\n")

content = tokenizer.decode(
    output_ids[index:],
    skip_special_tokens=True,
).strip("\n")


# ============================================================
# 9. 打印结果
# ============================================================

print("\n" + "=" * 60)
print("THINKING:")
print(thinking_content)

print("\n" + "=" * 60)
print("CONTENT:")
print(content)