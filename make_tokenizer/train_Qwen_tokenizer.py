from pathlib import Path

from tokenizers import Tokenizer, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.processors import ByteLevel as ByteLevelPostProcessor
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFC
from tokenizers.pre_tokenizers import Sequence, Split
from tokenizers import AddedToken

# ============================================================
# Configuration
# ============================================================

CORPUS_DIR = Path("./corpus")
OUTPUT_DIR = Path("./tokenizer")

VOCAB_SIZE = 151643

MIN_FREQUENCY = 2

# ============================================================
# Special Tokens
# ============================================================

SPECIAL_TOKENS = {
    "pad_token": "<|endoftext|>",
    "eos_token": "<|im_end|>",
    "bos_token": None,
    "unk_token": None,

    "additional_special_tokens": [
        "<|im_start|>",
    ],
}
_special_token_strings = []

for token in [
    SPECIAL_TOKENS["pad_token"],
    SPECIAL_TOKENS["eos_token"],
    *SPECIAL_TOKENS["additional_special_tokens"],
]:
    if token is not None and token not in _special_token_strings:
        _special_token_strings.append(token)

added_special_tokens = [
    AddedToken(
        token,
        special=True,
        normalized=False,
        single_word=False,
        lstrip=False,
        rstrip=False,
    )
    for token in _special_token_strings
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
        unk_token=None,
        fuse_unk=False,
        byte_fallback=False,
        continuing_subword_prefix="",
        end_of_word_suffix="",
    )
)

# ============================================================
# Normalizer
# ============================================================

tokenizer.normalizer = NFC()

# ============================================================
# Pre-tokenizer
# ============================================================
pattern = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)
tokenizer.pre_tokenizer = Sequence([
    Split(
        pattern=Regex(pattern),
        behavior="isolated",
        invert=False,
    ),
    ByteLevelPreTokenizer(
        add_prefix_space=False,
        use_regex=False,
        trim_offsets=False,
    ),
])

# ByteLevel post processor
tokenizer.post_processor = ByteLevelPostProcessor(
    add_prefix_space=False,
    trim_offsets=False,
    use_regex=False,
)
# ============================================================
# Decoder
# ============================================================

tokenizer.decoder = ByteLevelDecoder(
    add_prefix_space=False,
    trim_offsets=False,
)

# ============================================================
# Trainer
# ============================================================

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=MIN_FREQUENCY,

    special_tokens=[],

    show_progress=True,

    # Byte-level BPE
    initial_alphabet=ByteLevelPreTokenizer.alphabet(),
)

# ============================================================
# Train
# ============================================================

tokenizer.train(
    files=files,
    trainer=trainer,
)
# ============================================================
# Add special tokens
# ============================================================
base_vocab_size = tokenizer.get_vocab_size()
print(f"Base vocabulary size: {base_vocab_size}")
tokenizer.add_special_tokens(added_special_tokens)
print(
    f"Final vocabulary size: {tokenizer.get_vocab_size()}"
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
    "add_bos_token": False,
    "add_prefix_space": False,
    "clean_up_tokenization_spaces": False,
    "model_max_length": 32768,

    "tokenizer_class": "PreTrainedTokenizerFast",

    "bos_token": SPECIAL_TOKENS["bos_token"],
    "eos_token": SPECIAL_TOKENS["eos_token"],
    "pad_token": SPECIAL_TOKENS["pad_token"],
    "unk_token": SPECIAL_TOKENS["unk_token"],

    "additional_special_tokens": SPECIAL_TOKENS[
        "additional_special_tokens"
    ],

    "split_special_tokens": False,
    "chat_template": (
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\\n' }}"
        "{% endif %}"
    ),
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
