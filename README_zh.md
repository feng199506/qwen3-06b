# Model_01

与 Model_02 对齐的大模型训练流水线目录：

**Tokenizer → 预训练 (LM) → SFT → DPO → GRPO**

English: [README.md](README.md).

---

## 环境

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
pip install -r requirements.txt
# 离线示例：
# pip install --no-index --find-links "../packs" -r .\requirements.txt
```

---

## 目录结构

```text
Model_01/
├── corpus/                 # 预训练语料
├── configs/                # 模型配置
├── tokenizer/              # 训练得到的分词器（gitignore）
├── 01_make_tokenizer/      # 训练 / 测试分词器
├── 02_train_lang_model/    # 从零预训练
├── 03_train_sft/           # SFT（含 data/）
├── 04_train_dpo/           # DPO（含 data/）
├── 05_train_grpo/          # GRPO（含 data/）
├── lm_output/              # 预训练产物
├── sft_output/             # SFT 产物（原 qa_output）
├── dpo_output/             # DPO 产物
├── grpo_output/            # GRPO 产物
└── sentence_transformer_configs/  # GRPO 语义奖励用本地向量模型
```

---

## 训练流程

```bash
# 1）Tokenizer（推荐入口）
python 01_make_tokenizer/train_Qwen_tokenizer.py

# 2）预训练
python 02_train_lang_model/train01.py

# 3）SFT 数据 + 训练
python 03_train_sft/gen_data.py
python 03_train_sft/gen_data2.py
python 03_train_sft/train_qa_sft.py

# 4）DPO 数据 + 训练
python 04_train_dpo/gen_dpo_data.py
python 04_train_dpo/train_qa_dpo.py

# 5）GRPO 数据 + 训练
python 05_train_grpo/gen_grpo_data.py
python 05_train_grpo/train_grpo.py
```

### 测试

```bash
python 02_train_lang_model/test.py
python 03_train_sft/test_qa.py
python 04_train_dpo/test_qa_dpo.py
```

脚本使用 `Path(__file__)` 解析路径，可在仓库根目录运行。

---

## 旧目录对照

| 旧路径 | 新路径 |
|--------|--------|
| `make_tokenizer/` | `01_make_tokenizer/` |
| `make_tokenizer/corpus` | `corpus/` |
| `make_tokenizer/tokenizer` | `tokenizer/` |
| `train_l_model/` | `02_train_lang_model/` |
| `train_QA/`（SFT） | `03_train_sft/` |
| `train_QA/dpo_data` | `04_train_dpo/data/` |
| `output/` | `lm_output/` |
| `qa_output/` | `sft_output/` |

---

## 说明

- 对话测试请使用 `sft_output` / `dpo_output` / `grpo_output`。
- 正式预训练前请扩充 `corpus/`。
- GRPO 需要 `sentence_transformer_configs/` 计算语义相似度奖励。
