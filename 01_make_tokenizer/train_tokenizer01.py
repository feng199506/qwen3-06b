from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "corpus"
OUTPUT_DIR = BASE_DIR / "tokenizer"

VOCAB_SIZE = 32000

MIN_FREQUENCY = 2

SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
]


# ============================================================
# Prepare corpus files
# ============================================================

files = sorted(
    str(path)
    for path in CORPUS_DIR.rglob("*.txt")
    if path.is_file()
)

if not files:
    raise RuntimeError(
        f"No .txt files found in {CORPUS_DIR.resolve()}"
    )

print("=" * 60)
print("Corpus")
print("=" * 60)

print(f"Corpus directory : {CORPUS_DIR.resolve()}")
print(f"Text files       : {len(files)}")

for file in files[:10]:
    print(" -", file)

if len(files) > 10:
    print(f" ... and {len(files) - 10} more")

print()


# ============================================================
# Create tokenizer
# ============================================================

tokenizer = Tokenizer(
    BPE(
        unk_token="<|endoftext|>",
        byte_fallback=True,
    )
)


# ============================================================
# Normalizer
# ============================================================

tokenizer.normalizer = NFKC()


# ============================================================
# Pre-tokenizer
# ============================================================

tokenizer.pre_tokenizer = ByteLevel(
    add_prefix_space=False,
    use_regex=True,
)


# ============================================================
# Decoder
# ============================================================

tokenizer.decoder = ByteLevelDecoder()


# ============================================================
# Trainer
# ============================================================

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=MIN_FREQUENCY,

    special_tokens=SPECIAL_TOKENS,

    show_progress=True,

    # Byte-level BPE
    initial_alphabet=ByteLevel.alphabet(),
)


# ============================================================
# Train
# ============================================================

print("=" * 60)
print("Training tokenizer")
print("=" * 60)

print(f"Vocabulary size : {VOCAB_SIZE}")
print(f"Min frequency   : {MIN_FREQUENCY}")
print()

tokenizer.train(
    files=files,
    trainer=trainer,
)


# ============================================================
# Save tokenizer.json
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

tokenizer_json = OUTPUT_DIR / "tokenizer.json"

tokenizer.save(
    str(tokenizer_json)
)

print()
print("=" * 60)
print("Tokenizer trained")
print("=" * 60)

print(
    f"Vocabulary size: {tokenizer.get_vocab_size()}"
)

print(
    f"Saved: {tokenizer_json.resolve()}"
)


# ============================================================
# Export vocab.json + merges.txt
# ============================================================

model = tokenizer.model

if not isinstance(model, BPE):
    raise RuntimeError("Tokenizer model is not BPE")

vocab_path = OUTPUT_DIR / "vocab.json"
merges_path = OUTPUT_DIR / "merges.txt"

model.save(
    str(OUTPUT_DIR)
)

print()
print("Exported:")
print(" -", vocab_path)
print(" -", merges_path)


# ============================================================
# Create tokenizer_config.json
# ============================================================

import json

tokenizer_config = {
    "add_prefix_space": False,

    "clean_up_tokenization_spaces": False,

    "model_max_length": 32768,

    "tokenizer_class": "PreTrainedTokenizerFast",

    "bos_token": "<|im_start|>",

    "eos_token": "<|im_end|>",

    "unk_token": "<|endoftext|>",

    "pad_token": "<|endoftext|>",

    "additional_special_tokens": [
        "<|im_start|>",
        "<|im_end|>",
    ],
}


with open(
    OUTPUT_DIR / "tokenizer_config.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        tokenizer_config,
        f,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# Test tokenizer
# ============================================================

print()
print("=" * 60)
print("Testing")
print("=" * 60)


test_texts = [
    "Hello world!",
    "你好，世界！",
    "こんにちは世界",
    "机械设备故障诊断系统",
    "Python def hello(): print('hello')",
    "🤖🚀 AI Transformer",
]


for text in test_texts:

    encoded = tokenizer.encode(text)

    decoded = tokenizer.decode(
        encoded.ids
    )

    print()
    print("TEXT:")
    print(text)

    print("TOKENS:")
    print(encoded.tokens)

    print("IDS:")
    print(encoded.ids)

    print("DECODE:")
    print(decoded)

    if decoded != text:
        print("WARNING: round-trip mismatch!")

    else:
        print("OK")


print()
print("=" * 60)
print("DONE")
print("=" * 60)