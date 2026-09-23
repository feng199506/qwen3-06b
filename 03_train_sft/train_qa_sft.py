import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from trl import SFTTrainer, SFTConfig


# ============================================================
# 1. 路径
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "lm_output" / "final"
DATA_PATH = Path(__file__).resolve().parent / "data"
OUTPUT_PATH = BASE_DIR / "sft_output"


# ============================================================
# 2. 加载 tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    trust_remote_code=True,
)

print("Tokenizer loaded")
print("Tokenizer vocab_size:", tokenizer.vocab_size)
print("Tokenizer length:", len(tokenizer))
print("Chat template:", tokenizer.chat_template is not None)

# ============================================================
# 2.5 设置 SFT Training Chat Template
# ============================================================

CHAT_TEMPLATE = (
    "{% for message in messages %}"

    "{% if message['role'] == 'assistant' %}"

    "{{ '<|im_start|>assistant\\n' }}"

    "{% generation %}"
    "{{ message['content'] }}"

    "{{ '<|im_end|>\\n' }}"

    "{% endgeneration %}"

    "{% else %}"

    "{{ '<|im_start|>' + message['role'] + '\\n' "
    "+ message['content'] + '<|im_end|>\\n' }}"

    "{% endif %}"

    "{% endfor %}"

    "{% if add_generation_prompt %}"

    "{{ '<|im_start|>assistant\\n' }}"

    "{% endif %}"
)

tokenizer.chat_template = CHAT_TEMPLATE

print("SFT chat template configured.")
# ============================================================
# 3. 加载模型
# ============================================================

model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_PATH),
    trust_remote_code=True,
    dtype="auto",
)
# model.config.use_cache = False
print("Model loaded")
print("Model vocab_size:", model.config.vocab_size)


# ============================================================
# 4. 检查 tokenizer / model vocab
# ============================================================

assert len(tokenizer) == model.config.vocab_size, (
    f"Tokenizer/model vocab mismatch: "
    f"{len(tokenizer)} != {model.config.vocab_size}"
)

print("Vocab check: OK")


# ============================================================
# 5. 加载 QA 数据
# ============================================================

data_dir = Path(DATA_PATH)

json_files = list(data_dir.glob("*.json"))
jsonl_files = list(data_dir.glob("*.jsonl"))

data_files = json_files + jsonl_files

print("Found data files:")

for file in data_files:
    print("  ", file)

if not data_files:
    raise FileNotFoundError(
        f"No .json or .jsonl files found in: {DATA_PATH}"
    )

dataset = load_dataset(
    "json",
    data_files=[str(f) for f in data_files],
    split="train",
)
# dataset = dataset.select(range(10))
print("Dataset size:", len(dataset))
print("Dataset columns:", dataset.column_names)


# ============================================================
# 6. 检查 messages
# ============================================================

if "messages" not in dataset.column_names:
    raise ValueError(
        "Dataset must contain a 'messages' column."
    )

print()
print("=" * 80)
print("SAMPLE")
print("=" * 80)

print(dataset[0])


# ============================================================
# 7. 检查 chat template
# ============================================================

if tokenizer.chat_template is None:
    raise ValueError(
        "Tokenizer does not have a chat_template."
    )

# 测试一条数据
test_text = tokenizer.apply_chat_template(
    dataset[0]["messages"],
    tokenize=False,
    add_generation_prompt=False,
)

print()
print("=" * 80)
print("CHAT TEMPLATE TEST")
print("=" * 80)

print(test_text[:1000])


# ============================================================
# 8. 训练参数
# ============================================================

training_args = SFTConfig(
    output_dir=str(OUTPUT_PATH),

    num_train_epochs=10,

    per_device_train_batch_size=1,

    gradient_accumulation_steps=8,

    learning_rate=2e-5,

    max_length=1024,

    logging_steps=10,

    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,

    bf16=True,
    gradient_checkpointing=True,

    report_to="none",

    remove_unused_columns=False,
    assistant_only_loss=True
)


# ============================================================
# 9. SFT Trainer
# ============================================================

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=training_args,
)


# ============================================================
# 10. 开始训练
# ============================================================

print()
print("=" * 80)
print("START QA TRAINING")
print("=" * 80)

trainer.train()


# ============================================================
# 11. 保存 QA 模型
# ============================================================

trainer.save_model(str(OUTPUT_PATH))

tokenizer.save_pretrained(str(OUTPUT_PATH))

print()
print("=" * 80)
print("DONE")
print("=" * 80)

print("QA training finished")
print("Saved to:", OUTPUT_PATH)