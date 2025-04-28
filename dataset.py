from transformers import DataCollatorForLanguageModeling
from modelscope import AutoTokenizer
import json
from torch.utils.data import Dataset
from tqdm import tqdm

class InstructionDataset(Dataset):
    def __init__(self, tokenizer, data_path, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(data_path, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)
        
        self.processed_data = []
        for example in tqdm(self.raw_data, desc="Processing data"):
            formatted = self.format_instruction(example)
            if formatted:
                tokenized = self.tokenizer(
                    formatted,
                    truncation=True,
                    max_length=self.max_length,
                    padding=False,
                    return_tensors=None
                )
                self.processed_data.append(tokenized)

    @staticmethod
    def format_instruction(example):        
        messages = example.get("messages")
        if not isinstance(messages, list):
            return None
            
        conversation = []
        for message in messages:
            if message.get("role") == "user":
                conversation.append(f"<|im_start|>user\n{message.get('content', '')}<|im_end|>")
            elif message.get("role") == "assistant":
                conversation.append(f"<|im_start|>assistant\n{message.get('content', '')}<|im_end|>")
        return "\n".join(conversation) if conversation else None

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]
    
class DataCollator:
    @staticmethod
    def create(tokenizer):
        return DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<think>", "</think>",  "<answer>", "</answer>"]})
    
    # Load dataset
    dataset = InstructionDataset(tokenizer, "./data/train.json")
    data_collator = DataCollator.create(tokenizer)
    batch = [dataset[i] for i in range(2)]
    collated_batch = data_collator(batch)
    
    print("Collated batch keys:", collated_batch.keys())
    print("Input shape:", collated_batch["input_ids"].shape)
    print("Attention mask shape:", collated_batch["attention_mask"].shape)
    print("Labels shape:", collated_batch["labels"].shape)