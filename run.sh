#!/bin/bash

# 可调整的超参数

# cand_model_name = {
#     "Qwen/Qwen2.5-Math-1.5B-Instruct", 
#     "Qwen/Qwen2.5-Math-7B-Instruct"
#     "Qwen/Qwen2.5-Code-1.5B-Instruct"
#     "Qwen/Qwen2.5-Code-7B-Instruct"
# }

#!/bin/bash

# Configuration
MODEL_NAME="Qwen/Qwen2.5-Math-1.5B-Instruct"
DATA_PATH="./data/train.json"
OUTPUT_DIR="/root/autodl-tmp/model_outputs"  # Changed to avoid conflict with cache dir

# LoRA configuration
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.1
TARGET_MODULES="q_proj,v_proj"

# Training parameters
BATCH_SIZE=16
NUM_EPOCHS=3
LEARNING_RATE=5e-5
WEIGHT_DECAY=0.01
MAX_LENGTH=512

USE_FP16=true  # Will be handled as a flag
LOG_STEPS=10
SAVE_STRATEGY="epoch"
EVAL_STRATEGY="epoch"
SEED=42

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Optional: Activate virtual environment (uncomment and modify as needed)
# source /path/to/venv/bin/activate

# Check if train.py exists
if [ ! -f "train.py" ]; then
    echo "Error: train.py not found in current directory"
    exit 1
fi

# Run training
python train.py \
  --model_name "$MODEL_NAME" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT" \
  --target_modules "$TARGET_MODULES" \
  --batch_size "$BATCH_SIZE" \
  --num_epochs "$NUM_EPOCHS" \
  --learning_rate "$LEARNING_RATE" \
  --weight_decay "$WEIGHT_DECAY" \
  --max_length "$MAX_LENGTH" \
  $( [ "$USE_FP16" = "true" ] && echo "--use_fp16" ) \
  --log_steps "$LOG_STEPS" \
  --save_strategy "$SAVE_STRATEGY" \
  --eval_strategy "$EVAL_STRATEGY" \
  --seed "$SEED"

# Check exit status
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with error code $?"
    exit 1
fi