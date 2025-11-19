"""
Utility functions for logging, metrics, and visualization
"""

import matplotlib.pyplot as plt
import json
import os
from datetime import datetime


def plot_training_history(history, save_path=None):
    """Plot training and validation loss"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'train_loss' in history:
        ax.plot(history['steps'], history['train_loss'], label='Train Loss', alpha=0.8)
    if 'val_loss' in history:
        ax.plot(history['val_steps'], history['val_loss'], label='Validation Loss', alpha=0.8)
    
    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')
    ax.set_title('Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_training_log(log_data, log_dir="logs"):
    """Save training log to JSON file"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"training_log_{timestamp}.json")
    
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"Training log saved to {log_path}")


def calculate_perplexity(loss):
    """Calculate perplexity from loss"""
    import math
    return math.exp(loss)


def count_parameters(model):
    """Count total and trainable parameters in model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return total_params, trainable_params


def format_time(seconds):
    """Format seconds to human readable time"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class TrainingLogger:
    """Simple logger for training metrics"""
    
    def __init__(self, log_file="training.log"):
        self.log_file = log_file
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'steps': [],
            'val_steps': [],
            'learning_rates': []
        }
    
    def log_step(self, step, train_loss, lr):
        """Log training step"""
        self.history['steps'].append(step)
        self.history['train_loss'].append(train_loss)
        self.history['learning_rates'].append(lr)
        
        message = f"Step {step}: Loss={train_loss:.4f}, LR={lr:.6f}"
        self._write_log(message)
    
    def log_validation(self, step, val_loss):
        """Log validation"""
        self.history['val_steps'].append(step)
        self.history['val_loss'].append(val_loss)
        
        message = f"Validation at step {step}: Loss={val_loss:.4f}"
        self._write_log(message)
    
    def _write_log(self, message):
        """Write message to log file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
    
    def save_history(self, save_path="training_history.json"):
        """Save training history to JSON"""
        with open(save_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"Training history saved to {save_path}")
