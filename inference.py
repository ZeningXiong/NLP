from modelscope import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load base model and tokenizer
base_model_path = "/root/autodl-tmp/model/Qwen/Qwen2.5-Math-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.padding_side  = 'left'

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation = "flash_attention_2"
)

# 2. Load LoRA adapter and merge
lora_path = "./model_outputs"
peft_model = PeftModel.from_pretrained(model, lora_path, is_trainable=False)
merged_model = peft_model.merge_and_unload()

# 3. Test cases
test_cases = [
    "Write a Python function to calculate factorial",
    "Explain bubble sort algorithm",
    "How to reverse a string in JavaScript?",
    "What's the time complexity of binary search?"
]

# 4. Generate outputs with merged model
print("\n" + "="*50 + " Merged Model Outputs " + "="*50)
template = "<|im_start|>user\n{query}<|im_end|><im_start>assistant\n<im_start><think>"

for case in test_cases:
    case = template.format(query=case)
    inputs = tokenizer(case, return_tensors="pt").to(device)
    output = merged_model.generate(**inputs, max_new_tokens=512)
    
    print(f"\nTest Case: {case}")
    print(f"\nOutput:\n{tokenizer.decode(output[0], skip_special_tokens=False)}")
    print("\n" + "-"*80)