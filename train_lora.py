import argparse
import os
import logging
import json
from datetime import datetime
from uuid import uuid4
import torch
from peft import LoraConfig, get_peft_model
from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, Trainer, set_seed

from dataset import InstructionDataset, DataCollator
from utils import TrainingVisualizer, ModelSaver
from datetime import datetime

# Constants
MODEL_CACHE_DIR = "/root/autodl-tmp/model"
BASE_LOG_DIR = "./logs"
os.makedirs(BASE_LOG_DIR, exist_ok=True)

def setup_logging(run_dir):
    """Configure logging for the training process."""
    log_file = os.path.join(run_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments with improved validation."""
    parser = argparse.ArgumentParser(description="LoRA Instruction Fine-tuning")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for model outputs")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout rate")
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj", 
                       help="Comma-separated list of target modules for LoRA")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--log_steps", type=int, default=10, help="Logging frequency in steps")
    parser.add_argument("--save_strategy", type=str, default="epoch", 
                       choices=["epoch", "steps", "no"], help="Model saving strategy")
    parser.add_argument("--eval_strategy", type=str, default="epoch", 
                       choices=["epoch", "steps", "no"], help="Evaluation strategy")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.lora_r <= 0:
        raise ValueError("LoRA rank must be positive")
    if args.lora_dropout < 0 or args.lora_dropout > 1:
        raise ValueError("LoRA dropout must be between 0 and 1")
    
    return args

def create_run_directory(base_dir):
    """Create a unique directory for this training run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}_{uuid4().hex[:4]}"
    run_dir = os.path.join(base_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, timestamp

def save_hyperparameters(args, run_dir):
    """Save training hyperparameters to a JSON file."""
    params_path = os.path.join(run_dir, "hyperparameters.json")
    with open(params_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    return params_path

def setup_model(args, logger):
    """Initialize model and tokenizer with caching."""
    logger.info(f"Loading model: {args.model_name}")
    
    model_kwargs = {
        "cache_dir": MODEL_CACHE_DIR,
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "attn_implementation": "flash_attention_2"
    }
    
    try:
        # Check if model exists in cache
        if os.path.exists(os.path.join(MODEL_CACHE_DIR, args.model_name)):
            logger.info(f"Loading model from cache: {MODEL_CACHE_DIR}/{args.model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                os.path.join(MODEL_CACHE_DIR, args.model_name),
                **model_kwargs
            )
            tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(MODEL_CACHE_DIR, args.model_name)
            )
        else:
            logger.info("Downloading model and saving to cache")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                **model_kwargs
            )
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            tokenizer.padding_side = 'left'
            # Save to cache
            model.save_pretrained(os.path.join(MODEL_CACHE_DIR, args.model_name))
            tokenizer.save_pretrained(os.path.join(MODEL_CACHE_DIR, args.model_name))
            
        # Configure tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
            
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

def setup_lora(model, args, logger):
    """Configure and apply LoRA to the model."""
    logger.info("Configuring LoRA...")
    try:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules.split(","),
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        logger.info("LoRA configuration applied successfully")
        model.print_trainable_parameters()
        return model
    except Exception as e:
        logger.error(f"Failed to setup LoRA: {str(e)}")
        raise

def main():
    """Main training function."""
    args = parse_args()
    set_seed(args.seed)
    
    run_dir, timestamp = create_run_directory(BASE_LOG_DIR)
    logger = setup_logging(run_dir)
    
    params_path = save_hyperparameters(args, run_dir)
    logger.info(f"Saved hyperparameters to {params_path}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    
    try:
        model, tokenizer = setup_model(args, logger)
        tokenizer.padding_side  = 'left'

        model = setup_lora(model, args, logger)
        
        # Initialize visualizer with run directory
        visualizer = TrainingVisualizer(run_dir)
        
        # Load dataset
        logger.info("Loading dataset...")
        dataset = InstructionDataset(tokenizer, args.data_path, args.max_length)
        data_collator = DataCollator.create(tokenizer)
        
        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            logging_dir=run_dir,
            logging_steps=args.log_steps,
            fp16=True,
            push_to_hub=False,
            run_name=f"lora-finetune-{uuid4().hex[:8]}",
            label_names=["labels"],
            gradient_accumulation_steps=4,
            save_strategy=args.save_strategy,
            remove_unused_columns=False,
            load_best_model_at_end=False
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        ModelSaver.save(model, tokenizer, args.output_dir, timestamp)
        logger.info(f"Model saved to {args.output_dir}")
        
        plot_path = visualizer.plot_loss(run_dir)
        visualizer.close()
        logger.info(f"Loss plot saved to {plot_path}")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    finally:
        # Clean up resources
        if 'visualizer' in locals():
            visualizer.close()

if __name__ == "__main__":
    main()