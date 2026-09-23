# Model_01

LLM training pipeline aligned with the same layout as Model_02:

**Tokenizer → Pretrain (LM) → SFT → DPO → GRPO**

中文说明见 [README_zh.md](README_zh.md).

---

## Setup

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
pip install -r requirements.txt
# or offline:
# pip install --no-index --find-links "../packs" -r .\requirements.txt
```

---

## Layout

```text
Model_01/
├── corpus/                 # Pretrain text
├── configs/                # Model config (+ optional HF assets)
├── tokenizer/              # Trained tokenizer (gitignored)
├── 01_make_tokenizer/      # Train / test tokenizer
├── 02_train_lang_model/    # Causal LM from scratch
├── 03_train_sft/           # SFT (+ data/)
├── 04_train_dpo/           # DPO (+ data/)
├── 05_train_grpo/          # GRPO (+ data/)
├── lm_output/              # Pretrain output
├── sft_output/             # SFT output (was qa_output)
├── dpo_output/             # DPO output
├── grpo_output/            # GRPO output
└── sentence_transformer_configs/  # Local embedding model for GRPO rewards
```

---

## Pipeline

```bash
# 1) Tokenizer (recommended entry)
python 01_make_tokenizer/train_Qwen_tokenizer.py

# 2) Pretrain
python 02_train_lang_model/train01.py

# 3) SFT data + train
python 03_train_sft/gen_data.py
python 03_train_sft/gen_data2.py
python 03_train_sft/train_qa_sft.py

# 4) DPO data + train
python 04_train_dpo/gen_dpo_data.py
python 04_train_dpo/train_qa_dpo.py

# 5) GRPO data + train
python 05_train_grpo/gen_grpo_data.py
python 05_train_grpo/train_grpo.py
```

### Tests

```bash
python 02_train_lang_model/test.py
python 03_train_sft/test_qa.py
python 04_train_dpo/test_qa_dpo.py
```

Scripts resolve paths via `Path(__file__)`, so they can be run from the repo root.

---

## Name mapping (old → new)

| Old | New |
|-----|-----|
| `make_tokenizer/` | `01_make_tokenizer/` |
| `make_tokenizer/corpus` | `corpus/` |
| `make_tokenizer/tokenizer` | `tokenizer/` |
| `train_l_model/` | `02_train_lang_model/` |
| `train_QA/` (SFT) | `03_train_sft/` |
| `train_QA/dpo_data` | `04_train_dpo/data/` |
| `output/` | `lm_output/` |
| `qa_output/` | `sft_output/` |

---

## Notes

- Prefer testing chat models under `sft_output` / `dpo_output` / `grpo_output`.
- Expand `corpus/` before serious pretraining.
- GRPO needs `sentence_transformer_configs/` for the semantic reward.
