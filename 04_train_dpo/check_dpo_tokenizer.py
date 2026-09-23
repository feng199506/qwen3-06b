from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer


# ============================================================
# 1. 路径
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TOKENIZER_PATH = BASE_DIR / "tokenizer"
DPO_DATA_PATH = BASE_DIR / "04_train_dpo" / "data" / "dpo_train.jsonl"


# ============================================================
# 2. 加载 tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    str(TOKENIZER_PATH),
    trust_remote_code=True,
)

print("=" * 80)
print("Tokenizer")
print("=" * 80)

print("tokenizer.vocab_size =", tokenizer.vocab_size)
print("len(tokenizer)       =", len(tokenizer))
print("pad_token            =", repr(tokenizer.pad_token))
print("pad_token_id         =", tokenizer.pad_token_id)
print("eos_token            =", repr(tokenizer.eos_token))
print("eos_token_id         =", tokenizer.eos_token_id)

print()


# ============================================================
# 3. 加载 DPO 数据
# ============================================================

dataset = load_dataset(
    "json",
    data_files=str(DPO_DATA_PATH),
    split="train",
)

print("=" * 80)
print("Dataset")
print("=" * 80)

print("size =", len(dataset))
print("columns =", dataset.column_names)

print()


# ============================================================
# 4. 检查 token prefix
# ============================================================

bad_count = 0

for i, row in enumerate(dataset):

    prompt = row["prompt"]
    chosen = row["chosen"]
    rejected = row["rejected"]

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
    )["input_ids"]

    prompt_chosen_ids = tokenizer(
        prompt + chosen,
        add_special_tokens=True,
    )["input_ids"]

    prompt_rejected_ids = tokenizer(
        prompt + rejected,
        add_special_tokens=True,
    )["input_ids"]

    chosen_ok = (
        prompt_chosen_ids[:len(prompt_ids)]
        == prompt_ids
    )

    rejected_ok = (
        prompt_rejected_ids[:len(prompt_ids)]
        == prompt_ids
    )

    if not chosen_ok or not rejected_ok:

        bad_count += 1

        print("=" * 80)
        print(f"BAD EXAMPLE #{i}")
        print("=" * 80)

        print("\nPROMPT:")
        print(repr(prompt))

        print("\nCHOSEN:")
        print(repr(chosen))

        print("\nREJECTED:")
        print(repr(rejected))

        print("\nPROMPT IDS:")
        print(prompt_ids)

        print("\nPROMPT+CHOSEN IDS:")
        print(prompt_chosen_ids)

        print("\nPROMPT+REJECTED IDS:")
        print(prompt_rejected_ids)

        print("\nchosen prefix:")
        print(prompt_chosen_ids[:len(prompt_ids)])

        print("\nrejected prefix:")
        print(prompt_rejected_ids[:len(prompt_ids)])

        # 找第一个不同的位置
        min_len = min(
            len(prompt_ids),
            len(prompt_chosen_ids),
        )

        print("\nFIRST CHOSEN DIFFERENCE:")

        for j in range(min_len):
            if prompt_ids[j] != prompt_chosen_ids[j]:
                print("position =", j)
                print("prompt token =", prompt_ids[j])
                print("combined token =", prompt_chosen_ids[j])

                print(
                    "prompt decoded =",
                    repr(tokenizer.decode(prompt_ids[:j + 5]))
                )

                print(
                    "combined decoded =",
                    repr(tokenizer.decode(prompt_chosen_ids[:j + 5]))
                )
                break

        min_len = min(
            len(prompt_ids),
            len(prompt_rejected_ids),
        )

        print("\nFIRST REJECTED DIFFERENCE:")

        for j in range(min_len):
            if prompt_ids[j] != prompt_rejected_ids[j]:
                print("position =", j)
                print("prompt token =", prompt_ids[j])
                print("combined token =", prompt_rejected_ids[j])

                print(
                    "prompt decoded =",
                    repr(tokenizer.decode(prompt_ids[:j + 5]))
                )

                print(
                    "combined decoded =",
                    repr(tokenizer.decode(prompt_rejected_ids[:j + 5]))
                )
                break

        print()

        # 只打印前 10 条坏数据
        if bad_count >= 10:
            break


print("=" * 80)
print("RESULT")
print("=" * 80)

print("bad_count =", bad_count)
print("checked   =", min(len(dataset), i + 1))