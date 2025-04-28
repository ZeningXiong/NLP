import subprocess
import os
from pathlib import Path



# Configuration
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B-Instruct"
DATA_PATH = "./data/training_code.json"
OUTPUT_DIR = "/root/autodl-tmp/model_outputs"  # Changed to avoid conflict with cache dir

# LoRA configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
TARGET_MODULES = "q_proj,v_proj,o_proj,v_proj"

# Training parameters
BATCH_SIZE = 2
NUM_EPOCHS = 3
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
MAX_LENGTH = 4096

USE_FP16 = False  # Will be handled as a flag
LOG_STEPS = 10
SAVE_STRATEGY = "epoch"
EVAL_STRATEGY = "epoch"
SEED = 42

# Ensure output directory exists
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)



# Build the command
command = [
    "python", "train_lora.py",
    "--model_name", MODEL_NAME,
    "--data_path", DATA_PATH,
    "--output_dir", OUTPUT_DIR,
    "--lora_r", str(LORA_R),
    "--lora_alpha", str(LORA_ALPHA),
    "--lora_dropout", str(LORA_DROPOUT),
    "--target_modules", TARGET_MODULES,
    "--batch_size", str(BATCH_SIZE),
    "--num_epochs", str(NUM_EPOCHS),
    "--learning_rate", str(LEARNING_RATE),
    "--weight_decay", str(WEIGHT_DECAY),
    "--max_length", str(MAX_LENGTH),
    "--log_steps", str(LOG_STEPS),
    "--save_strategy", SAVE_STRATEGY,
    "--eval_strategy", EVAL_STRATEGY,
    "--seed", str(SEED)
]

# Add the --use_fp16 flag if needed
if USE_FP16:
    command.append("--use_fp16")

# Run training
try:
    print("Executing command:", " ".join(command))
    result = subprocess.run(command, check=True)
    print("Training completed successfully!")
except subprocess.CalledProcessError as e:
    print(f"Training failed with error code {e.returncode}")
    exit(1)