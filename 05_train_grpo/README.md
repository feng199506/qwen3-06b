# GRPO

Scripts ported from Model_02.

## Usage

```bash
# Build dataset from SFT JSON/JSONL
python 05_train_grpo/gen_grpo_data.py

# Train (needs sft_output + sentence_transformer_configs/)
python 05_train_grpo/train_grpo.py
```

## Paths

| Item | Path |
|------|------|
| Base model | `sft_output/` (or switch to `dpo_output/` in script) |
| Data | `05_train_grpo/data/grpo_train.jsonl` |
| Semantic reward model | `sentence_transformer_configs/` |
| Output | `grpo_output/` |

Reward = ROUGE-L + semantic similarity + keyword overlap.
