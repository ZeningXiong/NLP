from modelscope import AutoTokenizer

model_name = "Qwen/Qwen2.5-Math-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "<think><answer></think></answer>"

# 编码后解码（验证是否能还原）
tokenizer.add_special_tokens({"additional_special_tokens": ["<think>", "<answer>","</answer>","</think>"]})
input_ids = tokenizer.encode(text)
print(input_ids)
decoded_text = tokenizer.decode(input_ids)
print("解码还原:", decoded_text)

