"""
Main training script for GPT model
"""

import torch
from torch.utils.data import DataLoader
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from configs.config import ModelConfig, TrainingConfig, DataConfig
from src.model.gpt import GPTModel
from src.data.dataset import TextDataset, collate_fn
from src.data.tokenizer import SimpleTokenizer
from src.data.download_dataset import load_texts
from src.training.trainer import Trainer
from src.utils.utils import count_parameters


def prepare_data():
    """Prepare datasets and tokenizer"""
    print("Preparing data...")
    
    # Check if processed data exists
    train_file = os.path.join(DataConfig.processed_data_dir, "train.txt")
    val_file = os.path.join(DataConfig.processed_data_dir, "val.txt")
    
    if not os.path.exists(train_file) or not os.path.exists(val_file):
        print("Processed data not found. Please run: python src/data/download_dataset.py")
        sys.exit(1)
    
    # Load texts
    print("Loading texts...")
    train_texts = load_texts(train_file)
    val_texts = load_texts(val_file)
    
    print(f"Loaded {len(train_texts)} training samples")
    print(f"Loaded {len(val_texts)} validation samples")
    
    # Initialize or load tokenizer
    tokenizer = SimpleTokenizer(vocab_size=DataConfig.vocab_size)
    tokenizer_path = DataConfig.tokenizer_dir
    
    if os.path.exists(os.path.join(tokenizer_path, "tokenizer.json")):
        print("Loading existing tokenizer...")
        tokenizer.load(tokenizer_path)
    else:
        print("Training new tokenizer...")
        tokenizer.train(train_texts, save_path=tokenizer_path)
    
    # Update model config with actual vocab size
    ModelConfig.vocab_size = tokenizer.get_vocab_size()
    print(f"Vocabulary size: {ModelConfig.vocab_size}")
    
    return train_texts, val_texts, tokenizer


def create_dataloaders(train_texts, val_texts, tokenizer):
    """Create PyTorch DataLoaders"""
    print("Creating datasets...")
    
    train_dataset = TextDataset(
        train_texts,
        tokenizer,
        max_length=ModelConfig.max_seq_length
    )
    
    val_dataset = TextDataset(
        val_texts,
        tokenizer,
        max_length=ModelConfig.max_seq_length
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=TrainingConfig.batch_size,
        shuffle=DataConfig.shuffle,
        num_workers=DataConfig.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=TrainingConfig.batch_size,
        shuffle=False,
        num_workers=DataConfig.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader


def main():
    """Main training function"""
    print("="*60)
    print("GPT Model Training")
    print("="*60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Prepare data
    train_texts, val_texts, tokenizer = prepare_data()
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(train_texts, val_texts, tokenizer)
    
    # Initialize model
    print("\nInitializing model...")
    model = GPTModel(ModelConfig)
    
    # Count parameters
    count_parameters(model)
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=TrainingConfig,
        tokenizer=tokenizer
    )
    
    # Start training
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60 + "\n")
    
    trainer.train()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Model checkpoints saved to: {TrainingConfig.checkpoint_dir}")


if __name__ == "__main__":
    main()
