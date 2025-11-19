"""
Training utilities and functions
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
import time
import os
from pathlib import Path


class Trainer:
    """Trainer class for GPT model"""
    
    def __init__(self, model, train_loader, val_loader, config, tokenizer):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.tokenizer = tokenizer
        
        # Setup device
        self.device = self._get_device()
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Setup learning rate scheduler
        total_steps = min(config.max_steps, len(train_loader) * config.max_epochs)
        if config.lr_scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)
        else:
            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=total_steps
            )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    def _get_device(self):
        """Get available device (MPS for Mac M4, CUDA, or CPU)"""
        if self.config.device == "mps" and torch.backends.mps.is_available():
            print("Using MPS (Metal Performance Shaders) for acceleration")
            return torch.device("mps")
        elif torch.cuda.is_available():
            print("Using CUDA for acceleration")
            return torch.device("cuda")
        else:
            print("Using CPU")
            return torch.device("cpu")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask, labels)
            loss = outputs['loss']
            
            # Backward pass
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            # Logging
            if self.global_step % self.config.log_every_n_steps == 0:
                avg_loss = total_loss / num_batches
                lr = self.scheduler.get_last_lr()[0]
                print(f"Step {self.global_step}, Epoch {self.epoch}, "
                      f"Loss: {avg_loss:.4f}, LR: {lr:.6f}")
            
            # Evaluation
            if self.global_step % self.config.eval_every_n_steps == 0:
                val_loss = self.evaluate()
                print(f"Validation Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best_model.pt")
                    print(f"New best model saved! Val Loss: {val_loss:.4f}")
                
                self.model.train()
            
            # Checkpointing
            if self.global_step % self.config.save_every_n_steps == 0:
                self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")
            
            # Max steps check
            if self.global_step >= self.config.max_steps:
                break
        
        return total_loss / num_batches
    
    def evaluate(self):
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs['loss']
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        print(f"Total epochs: {self.config.max_epochs}")
        print(f"Max steps: {self.config.max_steps}")
        print(f"Device: {self.device}")
        
        start_time = time.time()
        
        for epoch in range(self.config.max_epochs):
            self.epoch = epoch
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{self.config.max_epochs}")
            print(f"{'='*50}")
            
            train_loss = self.train_epoch()
            print(f"Epoch {epoch + 1} - Average Train Loss: {train_loss:.4f}")
            
            # Evaluate at end of epoch
            val_loss = self.evaluate()
            print(f"Epoch {epoch + 1} - Validation Loss: {val_loss:.4f}")
            
            if self.global_step >= self.config.max_steps:
                print("Reached maximum steps!")
                break
        
        total_time = time.time() - start_time
        print(f"\nTraining complete! Total time: {total_time/3600:.2f} hours")
        
        # Save final model
        self.save_checkpoint("final_model.pt")
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.config.checkpoint_dir, filename)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from step {self.global_step}, epoch {self.epoch}")
