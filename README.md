1. install python 3.10.10
2. install virtual environment and activate
    python -m venv .venv
    ./.venv/Scripts/activate
3. install environment
    pip install --no-index --find-links "../packs" -r .\requirements.txt 
4. train tokenizer
    cd make_tokenizer
    python train_Qwen_tokenizer.py
5. train language model
    cd train_l_model
    python train01.py
6. generate sft data
    cd train_QA
    pyhon gen_data.py
    python gen_data2.py
7. sft train
    python train_qa_sft.py
8. generate dpo data
    python gen_dpo_data.py
9. dpo train
    python train_qa_dpo.py

