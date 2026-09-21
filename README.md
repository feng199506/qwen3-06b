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
6. train QA model
cd train_QA
python train_qa.py