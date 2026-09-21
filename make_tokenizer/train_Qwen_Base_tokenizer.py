from pathlib import Path
import json

from tokenizers import (
    Tokenizer,
    Regex,
    AddedToken,
)

from tokenizers.models import BPE

from tokenizers.trainers import BpeTrainer

from tokenizers.pre_tokenizers import (
    ByteLevel as ByteLevelPreTokenizer,
    Sequence,
    Split,
)

from tokenizers.processors import (
    ByteLevel as ByteLevelPostProcessor,
)

from tokenizers.decoders import (
    ByteLevel as ByteLevelDecoder,
)

from tokenizers.normalizers import NFC


# ============================================================
# Configuration
# ============================================================

CORPUS_DIR = Path("./corpus")

OUTPUT_DIR = Path("./tokenizer")


# ============================================================
# IMPORTANT
#
# 这里不再要求：
#
#     BPE vocab = 151643
#
# BPE 最终 vocab 由你的实际语料决定。
#
# 例如：
#
#     实际 BPE vocab = 9361
#
# 那么：
#
#     <|endoftext|> = 9361
#     <|im_start|>  = 9362
#     ...
#     </think>      = 9386
#
# 最终：
#
#     vocab = 9387
# ============================================================

VOCAB_SIZE = 151643

MIN_FREQUENCY = 2


# ============================================================
# Qwen3 Added Tokens
#
# 注意：
#
# 这些 token 不再强制使用 Qwen3 原始的
# 151643 ~ 151668 ID。
#
# 而是：
#
#     base_vocab_size
#     ↓
#     依次追加
#
# ============================================================


# ------------------------------------------------------------
# tokenizer.json 中 special=true
# ------------------------------------------------------------

SPECIAL_TRUE_TOKENS = [

    "<|endoftext|>",

    "<|im_start|>",

    "<|im_end|>",

    "<|object_ref_start|>",

    "<|object_ref_end|>",

    "<|box_start|>",

    "<|box_end|>",

    "<|quad_start|>",

    "<|quad_end|>",

    "<|vision_start|>",

    "<|vision_end|>",

    "<|vision_pad|>",

    "<|image_pad|>",

    "<|video_pad|>",
]


# ------------------------------------------------------------
# tokenizer.json 中 special=false
# ------------------------------------------------------------

SPECIAL_FALSE_TOKENS = [

    "<tool_call>",

    "</tool_call>",

    "<|fim_prefix|>",

    "<|fim_middle|>",

    "<|fim_suffix|>",

    "<|fim_pad|>",

    "<|repo_name|>",

    "<|file_sep|>",

    "<tool_response>",

    "</tool_response>",

    "<think>",

    "</think>",
]


# ============================================================
# Transformers additional_special_tokens
#
# <|endoftext|> 不放这里
# ============================================================

ADDITIONAL_SPECIAL_TOKENS = [

    "<|im_start|>",

    "<|im_end|>",

    "<|object_ref_start|>",

    "<|object_ref_end|>",

    "<|box_start|>",

    "<|box_end|>",

    "<|quad_start|>",

    "<|quad_end|>",

    "<|vision_start|>",

    "<|vision_end|>",

    "<|vision_pad|>",

    "<|image_pad|>",

    "<|video_pad|>",
]


# ============================================================
# Sanity Check
# ============================================================

assert len(SPECIAL_TRUE_TOKENS) == 14

assert len(SPECIAL_FALSE_TOKENS) == 12

ALL_QWEN3_TOKENS = (
    SPECIAL_TRUE_TOKENS
    + SPECIAL_FALSE_TOKENS
)

assert len(ALL_QWEN3_TOKENS) == 26

assert len(set(ALL_QWEN3_TOKENS)) == 26


# ============================================================
# Prepare corpus
# ============================================================

files = sorted(
    str(path)
    for path in CORPUS_DIR.rglob("*.txt")
    if path.is_file()
)


if not files:

    raise RuntimeError(
        f"No .txt files found in "
        f"{CORPUS_DIR.resolve()}"
    )


print()
print("=" * 70)
print("Corpus")
print("=" * 70)

print(
    f"Corpus directory : "
    f"{CORPUS_DIR.resolve()}"
)

print(
    f"Text files       : "
    f"{len(files)}"
)


for file in files[:10]:

    print(
        " -",
        file
    )


if len(files) > 10:

    print(
        f" ... and "
        f"{len(files) - 10} more"
    )


# ============================================================
# Create BPE tokenizer
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


# ============================================================
# Post processor
# ============================================================

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
# BPE Trainer
# ============================================================

trainer = BpeTrainer(

    # 这里仍然可以设置 151643。
    #
    # 意思是：
    #     最多训练 151643
    #
    # 而不是：
    #     必须训练到 151643
    vocab_size=VOCAB_SIZE,

    min_frequency=MIN_FREQUENCY,

    # 非常重要：
    # special token 不参与 BPE 训练。
    special_tokens=[],

    show_progress=True,

    # Byte-level BPE
    initial_alphabet=(
        ByteLevelPreTokenizer.alphabet()
    ),
)


# ============================================================
# Train BPE
# ============================================================

print()
print("=" * 70)
print("Training BPE")
print("=" * 70)

tokenizer.train(

    files=files,

    trainer=trainer,
)


# ============================================================
# Check actual BPE vocab
# ============================================================

base_vocab_size = (
    tokenizer.get_vocab_size()
)


print()
print("=" * 70)
print("BPE vocabulary")
print("=" * 70)

print(
    f"Requested BPE vocab : "
    f"{VOCAB_SIZE}"
)

print(
    f"Actual BPE vocab    : "
    f"{base_vocab_size}"
)


# ============================================================
# IMPORTANT
#
# 这里不再检查：
#
#     <|endoftext|> == 151643
#
# 因为你的 BPE vocab 是实际训练结果。
#
# ============================================================


# ============================================================
# Create AddedToken objects
# ============================================================

special_true_added_tokens = [

    AddedToken(

        token,

        single_word=False,

        lstrip=False,

        rstrip=False,

        normalized=False,

        special=True,
    )

    for token in SPECIAL_TRUE_TOKENS
]


special_false_added_tokens = [

    AddedToken(

        token,

        single_word=False,

        lstrip=False,

        rstrip=False,

        normalized=False,

        special=False,
    )

    for token in SPECIAL_FALSE_TOKENS
]


# ============================================================
# Add special=true tokens FIRST
#
# ID:
#
#     base_vocab_size
#     ...
#
# ============================================================

print()
print("=" * 70)
print("Adding special=true tokens")
print("=" * 70)


added_special_count = (
    tokenizer.add_special_tokens(
        special_true_added_tokens
    )
)


print(
    f"Added special=true tokens: "
    f"{added_special_count}"
)


# ============================================================
# Add special=false tokens SECOND
#
# ID:
#
#     base_vocab_size + 14
#     ...
#
# ============================================================

print()
print("=" * 70)
print("Adding special=false tokens")
print("=" * 70)


added_normal_count = (
    tokenizer.add_tokens(
        special_false_added_tokens
    )
)


print(
    f"Added special=false tokens: "
    f"{added_normal_count}"
)


# ============================================================
# Check final vocab
# ============================================================

final_vocab_size = (
    tokenizer.get_vocab_size()
)


expected_final_vocab_size = (
    base_vocab_size + 26
)


print()
print("=" * 70)
print("Final vocabulary")
print("=" * 70)

print(
    f"Base vocab       : "
    f"{base_vocab_size}"
)

print(
    f"Added tokens     : "
    f"26"
)

print(
    f"Expected final   : "
    f"{expected_final_vocab_size}"
)

print(
    f"Actual final     : "
    f"{final_vocab_size}"
)


if final_vocab_size != expected_final_vocab_size:

    raise RuntimeError(

        "\n"
        "Final vocabulary size mismatch!\n"

        f"Base vocab: "
        f"{base_vocab_size}\n"

        f"Expected: "
        f"{expected_final_vocab_size}\n"

        f"Actual: "
        f"{final_vocab_size}\n"
    )


# ============================================================
# Verify exact IDs
#
# 这里才检查 token ID。
#
# 假设：
#
#     base_vocab_size = 9361
#
# 那么：
#
#     9361 <|endoftext|>
#     9362 <|im_start|>
#     ...
#     9386 </think>
#
# ============================================================

print()
print("=" * 70)
print("Checking token IDs")
print("=" * 70)


for expected_id, token in enumerate(

    ALL_QWEN3_TOKENS,

    start=base_vocab_size,
):

    actual_id = (
        tokenizer.token_to_id(token)
    )


    print(

        f"{actual_id:6d}  "
        f"{token}"
    )


    if actual_id != expected_id:

        raise RuntimeError(

            "\n"
            "Token ID mismatch!\n"

            f"Token:    {token}\n"

            f"Expected: {expected_id}\n"

            f"Actual:   {actual_id}"
        )


print()
print(
    "All token IDs are correct."
)


# ============================================================
# Verify special token count
# ============================================================

actual_added_count = (

    final_vocab_size
    - base_vocab_size
)


if actual_added_count != 26:

    raise RuntimeError(

        f"Expected 26 added tokens, "
        f"got {actual_added_count}"
    )


# ============================================================
# Output directory
# ============================================================

OUTPUT_DIR.mkdir(

    parents=True,

    exist_ok=True,
)


# ============================================================
# Save tokenizer.json
# ============================================================

tokenizer_json_path = (

    OUTPUT_DIR
    / "tokenizer.json"
)


tokenizer.save(

    str(tokenizer_json_path)
)


print()
print("=" * 70)
print("Saved tokenizer.json")
print("=" * 70)

print(
    tokenizer_json_path.resolve()
)


# ============================================================
# Export vocab.json + merges.txt
# ============================================================

model = tokenizer.model


if not isinstance(model, BPE):

    raise RuntimeError(
        "Tokenizer model is not BPE"
    )


model.save(
    str(OUTPUT_DIR)
)


vocab_path = (
    OUTPUT_DIR
    / "vocab.json"
)

merges_path = (
    OUTPUT_DIR
    / "merges.txt"
)


print()
print("=" * 70)
print("Exported BPE files")
print("=" * 70)

print(
    " -",
    vocab_path.resolve()
)

print(
    " -",
    merges_path.resolve()
)


# ============================================================
# Qwen3 Chat Template
# ============================================================

CHAT_TEMPLATE = r"""{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set content = message.content %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is defined and message.reasoning_content is not none %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in message.content %}
                {%- set content = message.content.split('</think>')[-1].lstrip('\n') %}
                {%- set reasoning_content = message.content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}"""


# ============================================================
# tokenizer_config.json
# ============================================================

all_added_tokens = SPECIAL_TRUE_TOKENS + SPECIAL_FALSE_TOKENS

added_tokens_decoder = {}

for token in all_added_tokens:
    token_id = tokenizer.token_to_id(token)

    if token_id is None:
        raise RuntimeError(
            f"Added token not found in tokenizer: {token}"
        )

    added_tokens_decoder[str(token_id)] = {
        "content": token,
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
        "special": token in SPECIAL_TRUE_TOKENS,
    }


tokenizer_config = {
    "add_bos_token": False,
    "add_prefix_space": False,

    "added_tokens_decoder": added_tokens_decoder,

    "additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS,

    "bos_token": None,
    "chat_template": CHAT_TEMPLATE,
    "clean_up_tokenization_spaces": False,
    "eos_token": "<|endoftext|>",
    "errors": "replace",
    "model_max_length": 131072,
    "pad_token": "<|endoftext|>",
    "split_special_tokens": False,
    "tokenizer_class": "Qwen2Tokenizer",
    "unk_token": None,
}


tokenizer_config_path = (

    OUTPUT_DIR
    / "tokenizer_config.json"
)
with open(
    tokenizer_config_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        tokenizer_config,
        f,
        ensure_ascii=False,
        indent=2,
    )

print()
print("=" * 70)
print("Saved tokenizer_config.json")
print("=" * 70)

print(
    tokenizer_config_path.resolve()
)
# ============================================================
# Force Qwen3-compatible ByteLevel decoder configuration
# ============================================================

with open(
    tokenizer_json_path,
    "r",
    encoding="utf-8",
) as f:
    tokenizer_json = json.load(f)


tokenizer_json["decoder"] = {
    "type": "ByteLevel",
    "add_prefix_space": False,
    "trim_offsets": False,
    "use_regex": False,
}


with open(
    tokenizer_json_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        tokenizer_json,
        f,
        ensure_ascii=False,
        indent=2,
    )


print()
print("=" * 70)
print("Fixed tokenizer.json decoder")
print("=" * 70)

print(
    json.dumps(
        tokenizer_json["decoder"],
        ensure_ascii=False,
        indent=2,
    )
)


# ============================================================
# Reload tokenizer.json
#
# 非常重要：
# 保存后重新加载，检查文件本身。
# ============================================================

print()
print("=" * 70)
print("Reloading tokenizer.json")
print("=" * 70)


reloaded_tokenizer = Tokenizer.from_file(

    str(tokenizer_json_path)
)


reloaded_vocab_size = (

    reloaded_tokenizer.get_vocab_size()
)


print(
    f"Reloaded vocab size = "
    f"{reloaded_vocab_size}"
)


if reloaded_vocab_size != final_vocab_size:

    raise RuntimeError(

        "Reloaded tokenizer vocab size "
        "does not match original tokenizer!"
    )


# ============================================================
# Verify IDs after reload
# ============================================================

print()
print("=" * 70)
print("Checking token IDs after reload")
print("=" * 70)


for expected_id, token in enumerate(

    ALL_QWEN3_TOKENS,

    start=base_vocab_size,
):

    actual_id = (

        reloaded_tokenizer
        .token_to_id(token)
    )


    print(

        f"{actual_id:6d}  "
        f"{token}"
    )


    if actual_id != expected_id:

        raise RuntimeError(

            "\n"
            "Reloaded token ID mismatch!\n"

            f"Token:    {token}\n"

            f"Expected: {expected_id}\n"

            f"Actual:   {actual_id}"
        )


print()
print(
    "Reloaded token IDs are correct."
)


# ============================================================
# Test encoding
# ============================================================

print()
print("=" * 70)
print("Testing tokenizer")
print("=" * 70)


test_texts = [

    "Hello world!",

    "你好，世界！",

    "こんにちは世界",

    "机械设备故障诊断系统",

    "Python def hello(): print('hello')",

    "🤖🚀 AI Transformer",
]


for text in test_texts:

    encoded = (
        reloaded_tokenizer
        .encode(text)
    )


    decoded = (
        reloaded_tokenizer
        .decode(encoded.ids)
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

        print(
            "WARNING: "
            "round-trip mismatch!"
        )

    else:

        print("OK")


# ============================================================
# Test special tokens
# ============================================================

print()
print("=" * 70)
print("Testing special tokens")
print("=" * 70)


special_test_text = (

    "<|im_start|>user\n"

    "你好\n"

    "<|im_end|>\n"

    "<|im_start|>assistant\n"
)


encoded = (

    reloaded_tokenizer
    .encode(special_test_text)
)


print("TEXT:")
print(special_test_text)


print()
print("TOKENS:")

print(
    encoded.tokens
)


print()
print("IDS:")

print(
    encoded.ids
)


print()
print("Decoded:")

print(

    reloaded_tokenizer
    .decode(encoded.ids)
)


# ============================================================
# Print final token mapping
# ============================================================

print()
print("=" * 70)
print("Final Qwen3-style token IDs")
print("=" * 70)


for token_id, token in zip(

    range(
        base_vocab_size,
        final_vocab_size,
    ),

    ALL_QWEN3_TOKENS,
):

    print(

        f"{token_id:6d}  "
        f"{token}"
    )


# ============================================================
# Final summary
# ============================================================

print()
print("=" * 70)
print("DONE")
print("=" * 70)


print(
    f"Requested BPE vocab : "
    f"{VOCAB_SIZE}"
)


print(
    f"Actual BPE vocab    : "
    f"{base_vocab_size}"
)


print(
    f"Added Qwen3 tokens  : "
    f"26"
)


print(
    f"Final vocab size    : "
    f"{final_vocab_size}"
)


print()
print(
    f"Tokenizer JSON      : "
    f"{tokenizer_json_path.resolve()}"
)


print(
    f"Vocab JSON          : "
    f"{vocab_path.resolve()}"
)


print(
    f"Merges TXT          : "
    f"{merges_path.resolve()}"
)


print(
    f"Config JSON         : "
    f"{tokenizer_config_path.resolve()}"
)


print()
print(
    "Tokenizer generation "
    "completed successfully."
)