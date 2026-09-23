from pathlib import Path
import json
import math
import time

from tokenizers import Tokenizer, Regex, AddedToken
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import (
    ByteLevel as ByteLevelPreTokenizer,
    Sequence,
    Split,
)
from tokenizers.processors import ByteLevel as ByteLevelPostProcessor
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFC


# ============================================================
# 配置
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "corpus"
OUTPUT_ROOT = Path(__file__).resolve().parent / "tokenizer_auto"

# 候选 vocab
CANDIDATE_VOCAB_SIZES = [
    1024,
    2048,
    4096,
    6144,
    8192,
    12288,
    16384,
    32768,
]

MIN_FREQUENCY = 2

# 用多少字符进行 evaluation
# None = 使用整个 corpus
MAX_EVAL_CHARS = 2_000_000

# 至少保留多少 special token
NUM_SPECIAL_TOKENS = 3


# ============================================================
# Special tokens
# ============================================================

special_tokens = [
    AddedToken(
        "<|endoftext|>",
        special=True,
        normalized=False,
    ),
    AddedToken(
        "<|im_start|>",
        special=True,
        normalized=False,
    ),
    AddedToken(
        "<|im_end|>",
        special=True,
        normalized=False,
    ),
]


# ============================================================
# Qwen3 Regex
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


# ============================================================
# 找 corpus
# ============================================================

files = [
    str(p)
    for p in CORPUS_DIR.rglob("*")
    if p.is_file()
]

if not files:
    raise RuntimeError(
        f"没有找到 corpus: {CORPUS_DIR.resolve()}"
    )

print("=" * 80)
print("CORPUS")
print("=" * 80)
print(f"文件数量: {len(files)}")


# ============================================================
# 创建 tokenizer
# ============================================================

def create_tokenizer():

    tokenizer = Tokenizer(
        BPE(
            unk_token=None,
            fuse_unk=False,
            byte_fallback=False,
        )
    )

    tokenizer.normalizer = NFC()

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

    tokenizer.post_processor = ByteLevelPostProcessor(
        add_prefix_space=False,
        trim_offsets=False,
        use_regex=False,
    )

    tokenizer.decoder = ByteLevelDecoder(
        add_prefix_space=False,
        trim_offsets=False,
    )

    return tokenizer


# ============================================================
# 准备 evaluation corpus
# ============================================================

def load_eval_text():

    chunks = []
    total_chars = 0

    for file in files:

        try:
            text = Path(file).read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            continue

        if not text:
            continue

        remaining = (
            MAX_EVAL_CHARS - total_chars
            if MAX_EVAL_CHARS is not None
            else None
        )

        if remaining is not None:

            if remaining <= 0:
                break

            text = text[:remaining]

        chunks.append(text)
        total_chars += len(text)

        if (
            MAX_EVAL_CHARS is not None
            and total_chars >= MAX_EVAL_CHARS
        ):
            break

    return chunks


eval_texts = load_eval_text()

eval_chars = sum(len(x) for x in eval_texts)

print(f"Evaluation chars: {eval_chars:,}")


# ============================================================
# 评估 tokenizer
# ============================================================

def evaluate_tokenizer(tokenizer):

    total_chars = 0
    total_tokens = 0

    for text in eval_texts:

        encoding = tokenizer.encode(text)

        total_chars += len(text)
        total_tokens += len(encoding.ids)

    if total_tokens == 0:
        return {
            "chars": 0,
            "tokens": 0,
            "chars_per_token": 0,
            "tokens_per_char": 0,
        }

    return {
        "chars": total_chars,
        "tokens": total_tokens,

        "chars_per_token":
            total_chars / total_tokens,

        "tokens_per_char":
            total_tokens / total_chars,
    }


# ============================================================
# 训练一个 tokenizer
# ============================================================

def train_tokenizer(target_vocab_size):

    print()
    print("=" * 80)
    print(f"TRAIN vocab_size={target_vocab_size}")
    print("=" * 80)

    start = time.time()

    tokenizer = create_tokenizer()

    trainer = BpeTrainer(
        vocab_size=target_vocab_size,
        min_frequency=MIN_FREQUENCY,

        # special token 后加
        special_tokens=[],

        show_progress=True,

        initial_alphabet=(
            ByteLevelPreTokenizer.alphabet()
        ),
    )

    tokenizer.train(
        files=files,
        trainer=trainer,
    )

    train_time = time.time() - start

    actual_vocab = tokenizer.get_vocab_size(
        with_added_tokens=False
    )

    metrics = evaluate_tokenizer(tokenizer)

    return {
        "target_vocab": target_vocab_size,
        "actual_vocab": actual_vocab,
        "train_time": train_time,
        **metrics,
    }


# ============================================================
# 训练所有候选 tokenizer
# ============================================================

results = []

for vocab_size in CANDIDATE_VOCAB_SIZES:

    result = train_tokenizer(vocab_size)

    results.append(result)

    print()
    print(
        f"Target={result['target_vocab']:,} "
        f"Actual={result['actual_vocab']:,} "
        f"Chars/Token={result['chars_per_token']:.4f}"
    )


# ============================================================
# 删除重复的实际 vocab
#
# 例如：
#
# 16384 -> 7944
# 32768 -> 7944
#
# 这两个没有必要同时参与推荐
# ============================================================

unique_results = []

seen_actual_vocab = set()

for r in results:

    actual = r["actual_vocab"]

    if actual in seen_actual_vocab:
        continue

    seen_actual_vocab.add(actual)
    unique_results.append(r)


# ============================================================
# 计算边际收益
# ============================================================

for i, r in enumerate(unique_results):

    if i == 0:

        r["gain"] = None
        r["gain_per_1000_vocab"] = None

        continue

    previous = unique_results[i - 1]

    vocab_gain = (
        r["actual_vocab"]
        - previous["actual_vocab"]
    )

    compression_gain = (
        r["chars_per_token"]
        - previous["chars_per_token"]
    )

    r["gain"] = compression_gain

    if vocab_gain > 0:

        r["gain_per_1000_vocab"] = (
            compression_gain
            / vocab_gain
            * 1000
        )

    else:

        r["gain_per_1000_vocab"] = 0


# ============================================================
# 自动选择
#
# 核心思想：
#
# 如果增加大量 vocab，
# chars/token 只增加很少，
# 就认为进入收益递减区。
# ============================================================

def recommend(results):

    if not results:
        raise RuntimeError("没有 tokenizer 结果")

    if len(results) == 1:
        return results[0]

    # --------------------------------------------------------
    # 方法：
    #
    # 找到一个 vocab：
    # 后续继续扩大 vocab，
    # 每增加 1000 个 token，
    # 压缩效率提升已经非常小。
    #
    # threshold 可以根据需求调整。
    # --------------------------------------------------------

    THRESHOLD = 0.005

    candidate = results[-1]

    for i in range(1, len(results)):

        r = results[i]

        gain = r["gain_per_1000_vocab"]

        if gain is None:
            continue

        if gain < THRESHOLD:

            candidate = r
            break

    # --------------------------------------------------------
    # 对非常小的 vocab 做保护
    # --------------------------------------------------------

    if candidate["actual_vocab"] < 2048:

        for r in results:

            if r["actual_vocab"] >= 4096:

                candidate = r
                break

    return candidate


recommended = recommend(unique_results)


# ============================================================
# 输出结果
# ============================================================

print()
print()
print("=" * 80)
print("TOKENIZER COMPARISON")
print("=" * 80)

print(
    f"{'Target':>10} "
    f"{'Actual':>10} "
    f"{'Tokens':>15} "
    f"{'Chars/Token':>15} "
    f"{'Tokens/Char':>15}"
)

print("-" * 80)

for r in unique_results:

    print(
        f"{r['target_vocab']:>10,} "
        f"{r['actual_vocab']:>10,} "
        f"{r['tokens']:>15,} "
        f"{r['chars_per_token']:>15.4f} "
        f"{r['tokens_per_char']:>15.4f}"
    )


# ============================================================
# 推荐结果
# ============================================================

recommended_base_vocab = (
    recommended["actual_vocab"]
)

recommended_final_vocab = (
    recommended_base_vocab
    + NUM_SPECIAL_TOKENS
)

print()
print()
print("=" * 80)
print("RECOMMENDATION")
print("=" * 80)

print(
    f"推荐 base vocab : "
    f"{recommended_base_vocab:,}"
)

print(
    f"Special tokens  : "
    f"{NUM_SPECIAL_TOKENS}"
)

print(
    f"推荐最终 vocab   : "
    f"{recommended_final_vocab:,}"
)

print(
    f"Chars/Token     : "
    f"{recommended['chars_per_token']:.4f}"
)

print(
    f"Tokens/Char     : "
    f"{recommended['tokens_per_char']:.4f}"
)


# ============================================================
# 给出解释
# ============================================================

print()
print("推荐理由：")

print(
    "1. vocab 已达到当前 corpus 的有效学习规模。"
)

print(
    "2. 继续增加 vocab 后，"
    "token 压缩效率提升明显变小。"
)

print(
    "3. 没有为了模仿 Qwen3 而强行制造大量低频 token。"
)

print(
    "4. Special tokens 会放在 base vocab 后面。"
)


# ============================================================
# 保存结果
# ============================================================

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

output = {
    "min_frequency": MIN_FREQUENCY,

    "evaluation_chars": eval_chars,

    "results": results,

    "unique_results": unique_results,

    "recommendation": {
        "base_vocab_size":
            recommended_base_vocab,

        "special_tokens":
            NUM_SPECIAL_TOKENS,

        "final_vocab_size":
            recommended_final_vocab,

        "chars_per_token":
            recommended["chars_per_token"],

        "tokens_per_char":
            recommended["tokens_per_char"],
    },
}

with open(
    OUTPUT_ROOT / "recommendation.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        output,
        f,
        ensure_ascii=False,
        indent=2,
    )

print()
print(
    f"结果保存到："
    f"{(OUTPUT_ROOT / 'recommendation.json').resolve()}"
)