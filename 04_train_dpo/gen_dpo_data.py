import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path
import json
import re
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


# ============================================================
# 1. 路径
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# SFT 后的模型
MODEL_PATH = BASE_DIR / "sft_output"

# 原来的 SFT 数据
DATA_PATH = BASE_DIR / "03_train_sft" / "data"

# DPO 数据输出目录
OUTPUT_DIR = BASE_DIR / "04_train_dpo" / "data"

# DPO 数据文件
OUTPUT_PATH = OUTPUT_DIR / "dpo_train.jsonl"


# ============================================================
# 2. 生成参数
# ============================================================

MAX_NEW_TOKENS = 256

TEMPERATURE = 0.5

TOP_P = 0.9
REPETITION_PENALTY = 1.05
# None = 全部数据
# 例如设置 10 可以先测试
MAX_SAMPLES = None

# 每多少条打印一次
PRINT_EVERY = 10


# ============================================================
# 3. 加载 tokenizer
# ============================================================

print("=" * 80)
print("Loading tokenizer")
print("=" * 80)

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    trust_remote_code=True,
)

print("Tokenizer vocab_size:", tokenizer.vocab_size)
print("Tokenizer length:", len(tokenizer))
print("Chat template:", tokenizer.chat_template is not None)


if tokenizer.chat_template is None:
    raise ValueError(
        "Tokenizer does not have a chat_template."
    )


# ============================================================
# 4. 加载 SFT 模型
# ============================================================

print()
print("=" * 80)
print("Loading SFT model")
print("=" * 80)

model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_PATH),
    trust_remote_code=True,
    dtype="auto",
)

model.eval()

print("Model vocab_size:", model.config.vocab_size)

assert len(tokenizer) == model.config.vocab_size, (
    f"Tokenizer/model vocab mismatch: "
    f"{len(tokenizer)} != {model.config.vocab_size}"
)

print("Vocab check: OK")


# ============================================================
# 5. 加载原始 SFT 数据
# ============================================================

print()
print("=" * 80)
print("Loading SFT dataset")
print("=" * 80)

data_dir = Path(DATA_PATH)

json_files = list(data_dir.glob("*.json"))
jsonl_files = list(data_dir.glob("*.jsonl"))

data_files = json_files + jsonl_files

if not data_files:
    raise FileNotFoundError(
        f"No .json or .jsonl files found in: {DATA_PATH}"
    )

print("Found data files:")

for file in data_files:
    print("  ", file)


dataset = load_dataset(
    "json",
    data_files=[str(f) for f in data_files],
    split="train",
)

print()
print("Dataset size:", len(dataset))
print("Dataset columns:", dataset.column_names)


# ============================================================
# 6. 检查 messages
# ============================================================

if "messages" not in dataset.column_names:
    raise ValueError(
        "SFT dataset must contain a 'messages' column."
    )


# ============================================================
# 7. 获取 user 问题
# ============================================================

def get_user_question(messages):
    for message in reversed(messages):
        if message["role"] == "user":
            return message["content"].strip()

    return ""


# ============================================================
# 8. 获取标准答案
# ============================================================

def get_reference_answer(messages):
    for message in reversed(messages):
        if message["role"] == "assistant":
            return message["content"].strip()

    return ""


# ============================================================
# 9. 生成 rejected
# ============================================================

@torch.inference_mode()
def generate_rejected(question):

    # --------------------------------------------------------
    # 只保留 user 问题
    # --------------------------------------------------------

    messages = [
        {
            "role": "user",
            "content": question,
        }
    ]

    # --------------------------------------------------------
    # 使用 chat template
    # --------------------------------------------------------

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    outputs = model.generate(
        **inputs,

        max_new_tokens=MAX_NEW_TOKENS,

        do_sample=True,

        temperature=TEMPERATURE,
        top_p=TOP_P,

        repetition_penalty=REPETITION_PENALTY,

        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # --------------------------------------------------------
    # 只取新生成部分
    # --------------------------------------------------------

    generated_tokens = outputs[
        0,
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return answer.strip()


# ============================================================
# 10. 清理 rejected
# ============================================================

def clean_answer(answer):

    if not answer:
        return ""

    answer = answer.strip()

    # --------------------------------------------------------
    # assistant 前缀
    # --------------------------------------------------------

    prefixes = [
        "assistant:",
        "Assistant:",
        "assistant：",
        "Assistant：",
    ]

    for prefix in prefixes:

        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()

    # --------------------------------------------------------
    # 防止继续生成下一轮
    # --------------------------------------------------------

    stop_markers = [
        "\nuser",
        "\nUser",
        "\n用户",
        "\n### User",
        "\n<|im_start|>user",
        "<|im_start|>user",
    ]

    for marker in stop_markers:

        if marker in answer:
            answer = answer.split(marker)[0].strip()

    # --------------------------------------------------------
    # 去除明显异常字符
    # --------------------------------------------------------

    answer = answer.replace("\x00", "")
    answer = answer.replace("\x01", "")
    answer = answer.replace("\x02", "")
    answer = answer.replace("\x03", "")
    answer = answer.replace("\x04", "")

    return answer.strip()


def valid_rejected(answer):

    if not answer:
        return False

    # 太短
    if len(answer) < 5:
        return False

    # replacement character
    replacement_count = answer.count("�")

    if replacement_count >= 2:
        return False

    # 控制字符
    bad_control_count = 0

    for ch in answer:
        code = ord(ch)

        if (
                code < 32
                and ch not in "\n\r\t"
        ):
            bad_control_count += 1

    if bad_control_count >= 2:
        return False

    # 连续重复字符，例如：
    # 哈哈哈哈哈哈哈哈哈
    if re.search(r"(.)\1{7,}", answer):
        return False
    return True
# ============================================================
# 11. 输出目录
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 12. 生成 DPO 数据
# ============================================================

print()
print("=" * 80)
print("GENERATING DPO DATA")
print("=" * 80)

total = len(dataset)

if MAX_SAMPLES is not None:
    total = min(
        total,
        MAX_SAMPLES,
    )

print("Total samples:", total)

written = 0
skipped = 0


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as f:

    for index in range(total):

        messages = dataset[index]["messages"]

        # ----------------------------------------------------
        # 原问题
        # ----------------------------------------------------

        question = get_user_question(messages)

        # ----------------------------------------------------
        # 原 SFT 标准答案
        #
        # 直接作为 chosen
        # ----------------------------------------------------

        chosen = get_reference_answer(messages)

        chosen = chosen.strip()

        # ----------------------------------------------------
        # SFT 模型重新生成回答
        #
        # 作为 rejected
        # ----------------------------------------------------

        rejected = generate_rejected(question)
        rejected = clean_answer(rejected)

        if not valid_rejected(rejected):
            skipped += 1
            continue

        # ----------------------------------------------------
        # 基础检查
        # ----------------------------------------------------

        if not question:
            skipped += 1
            continue

        if not chosen:
            skipped += 1
            continue

        if not rejected:
            skipped += 1
            continue

        # ----------------------------------------------------
        # 如果模型生成的答案和标准答案完全一样
        #
        # 没有 preference 信息，跳过
        # ----------------------------------------------------

        if rejected == chosen:
            skipped += 1
            continue

        # ----------------------------------------------------
        # DPO 数据
        # ----------------------------------------------------

        dpo_item = {
            "prompt": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
            "chosen": [
                {
                    "role": "assistant",
                    "content": chosen,
                }
            ],
            "rejected": [
                {
                    "role": "assistant",
                    "content": rejected,
                }
            ],
        }

        f.write(
            json.dumps(
                dpo_item,
                ensure_ascii=False,
            )
            + "\n"
        )

        written += 1

        # ----------------------------------------------------
        # 打印
        # ----------------------------------------------------

        if index % PRINT_EVERY == 0:

            print()
            print("-" * 80)

            print(
                f"Progress: {index + 1}/{total}"
            )

            print()
            print("PROMPT:")
            print(question)

            print()
            print("CHOSEN:")
            print(chosen)

            print()
            print("REJECTED:")
            print(rejected)


# ============================================================
# 13. 完成
# ============================================================

print()
print("=" * 80)
print("DONE")
print("=" * 80)

print("Input samples :", total)
print("Written samples:", written)
print("Skipped samples:", skipped)

print()
print("DPO dataset:")
print(OUTPUT_PATH)
