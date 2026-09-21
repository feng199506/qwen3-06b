from transformers import PreTrainedTokenizerFast


tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="./tokenizer/tokenizer.json"
)

tokenizer.add_special_tokens({
    "bos_token": "<|im_start|>",
    "eos_token": "<|im_end|>",
    "unk_token": "<|endoftext|>",
    "pad_token": "<|endoftext|>",
})


text = "你好，世界！Hello Transformer 🤖"

result = tokenizer(
    text,
    return_tensors=None,
)

print("TEXT:")
print(text)

print()

print("INPUT IDS:")
print(result["input_ids"])

print()

print("TOKENS:")
print(
    tokenizer.convert_ids_to_tokens(
        result["input_ids"]
    )
)

print()

print("DECODE:")

decoded = tokenizer.decode(
    result["input_ids"]
)

print(decoded)

print()

print(
    "Round trip:",
    decoded == text
)