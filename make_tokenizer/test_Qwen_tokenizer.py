from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./tokenizer")

print("=" * 60)
print("Tokenizer")
print("=" * 60)

print(tokenizer)

print()
print("=" * 60)
print("Type")
print("=" * 60)

print(type(tokenizer))
print(len(tokenizer))
print()
print("=" * 60)
print("Special Tokens")
print("=" * 60)

print("special_tokens_map:")
print(tokenizer.special_tokens_map)

print()
print("all_special_tokens:")
print(tokenizer.all_special_tokens)

print()
print("all_special_ids:")
print(tokenizer.all_special_ids)

print()
print("=" * 60)
print("Token IDs")
print("=" * 60)

for token in [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
]:
    print(
        f"{token:20s} -> "
        f"{tokenizer.convert_tokens_to_ids(token)}"
    )

texts = [
    "Hello world!",
    "你好，世界！",
    "机械设备故障诊断系统",
    "<|im_start|>user\n你好<|im_end|>",
]

for text in texts:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
    )

    decoded = tokenizer.decode(
        encoded["input_ids"],
        skip_special_tokens=False,
    )

    print("=" * 60)
    print("TEXT:", text)
    print("IDS:", encoded["input_ids"])
    print("TOKENS:", tokenizer.convert_ids_to_tokens(encoded["input_ids"]))
    print("DECODE:", decoded)

print("=" * 60)
messages = [
    {"role": "user", "content": "你好"},
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

print(text)