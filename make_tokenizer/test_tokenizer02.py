from transformers import PreTrainedTokenizerFast


tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="./tokenizer/tokenizer.json"
)

tests = [
    ("为什么测试集不能反复调参？", "如果不断根据测试集结果修改模型"),
    ("请简要介绍元组。", "元组是一种有序但不可变的序列"),
    ("请简要介绍监督学习。", "监督学习使用带标签的数据训练模型"),
    ("欠拟合主要解决什么问题？", "欠拟合表示模型没有充分学习数据中的规律"),
]

for prompt, answer in tests:

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
    )["input_ids"]

    answer_ids = tokenizer(
        answer,
        add_special_tokens=False,
    )["input_ids"]

    combined_ids = tokenizer(
        prompt + answer,
        add_special_tokens=False,
    )["input_ids"]

    manual_ids = prompt_ids + answer_ids

    print("=" * 80)

    print("PROMPT:", prompt)

    print("prompt:")
    print(prompt_ids)

    print("answer:")
    print(answer_ids)

    print("prompt + answer:")
    print(combined_ids)

    print("prompt + answer manually:")
    print(manual_ids)

    print()

    print(
        "prefix match:",
        combined_ids[:len(prompt_ids)] == prompt_ids
    )

    print(
        "combined == manual:",
        combined_ids == manual_ids
    )