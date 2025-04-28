import os
import logging
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import torch

class TrainingVisualizer:
    """A utility class for visualizing training metrics and saving loss plots."""
    
    def __init__(self, log_dir):
        """
        Initialize the TrainingVisualizer.
        
        Args:
            log_dir (str): Directory to store TensorBoard logs and loss plots.
        """
        self.log_dir = log_dir
        self.logger = logging.getLogger(__name__)
        
        os.makedirs(self.log_dir, exist_ok=True)
        
        try:
            self.writer = SummaryWriter(log_dir=self.log_dir)
            self.logger.info(f"Initialized TensorBoard writer at {self.log_dir}")
        except Exception as e:
            self.logger.error(f"Failed to initialize TensorBoard writer: {str(e)}")
            raise
            
    def plot_loss(self, output_dir):
        """
        Plot the training loss and save the plot to the output directory.
        
        Args:
            output_dir (str): Directory to save the loss plot.
            
        Returns:
            str: Path to the saved loss plot file.
        """
        try:
            plot_path = os.path.join(output_dir, "training_loss.png")

            plt.figure(figsize=(10, 6))
            plt.title("Training Loss")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.grid(True)
            plt.plot([0], [0], label="Loss (placeholder)")
            plt.legend()
            
            plt.savefig(plot_path)
            plt.close()
            
            self.logger.info(f"Saved loss plot to {plot_path}")
            return plot_path
            
        except Exception as e:
            self.logger.error(f"Failed to save loss plot: {str(e)}")
            raise
            
    def close(self):
        """Close the TensorBoard writer to flush all data."""
        try:
            self.writer.close()
            self.logger.info("Closed TensorBoard writer")
        except Exception as e:
            self.logger.error(f"Failed to close TensorBoard writer: {str(e)}")
            raise

import os
import logging
from datetime import datetime

class ModelSaver:
    """A utility class for saving models, tokenizers, and PEFT (LoRA) weights."""
    
    @staticmethod
    def save(model, tokenizer, output_dir, timestamp, save_full_model=False):
        """
        Save the model, tokenizer, and (optionally) PEFT (LoRA) weights.
        
        Args:
            model: The trained model (can be a PEFT model).
            tokenizer: The tokenizer to save.
            output_dir (str): Base directory to save the model.
            timestamp_format (str): Format for timestamp subfolder (set to None to disable).
            save_full_model (bool): If True, saves full `pytorch_model.bin` (default=False).
        """
        logger = logging.getLogger(__name__)
        
        try:
            output_dir = os.path.join(output_dir, f"model_{timestamp}")
            
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Saving model to: {output_dir}")

            if save_full_model:
                model_save_path = os.path.join(output_dir, "pytorch_model.bin")
                torch.save(model.state_dict(), model_save_path)
                logger.info(f"Saved model weights to {model_save_path}")
            
            model.config.save_pretrained(output_dir)
            logger.info(f"Saved model config to {output_dir}")

            tokenizer.save_pretrained(output_dir)
            logger.info(f"Saved tokenizer to {output_dir}")

            # Save PEFT (LoRA) weights if applicable
            if hasattr(model, "peft_config"):
                model.save_pretrained(output_dir)  # This saves adapter_model.bin
                logger.info(f"Saved PEFT (LoRA) adapter weights to {output_dir}")

        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise