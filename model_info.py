"""
Model information and inspection utility
"""

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from configs.config import ModelConfig
from src.model.gpt import GPTModel


def count_parameters(model):
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_number(num):
    """Format large numbers"""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)


def print_model_info():
    """Print detailed model information"""
    print("="*70)
    print("GPT Model Architecture Information")
    print("="*70)
    
    # Configuration
    print("\n📋 Model Configuration:")
    print(f"  • Vocabulary Size: {format_number(ModelConfig.vocab_size)}")
    print(f"  • Maximum Sequence Length: {ModelConfig.max_seq_length}")
    print(f"  • Number of Layers: {ModelConfig.n_layers}")
    print(f"  • Number of Attention Heads: {ModelConfig.n_heads}")
    print(f"  • Model Dimension (d_model): {ModelConfig.d_model}")
    print(f"  • Feed-Forward Dimension (d_ff): {ModelConfig.d_ff}")
    print(f"  • Dropout Rate: {ModelConfig.dropout}")
    print(f"  • Max Position Embeddings: {ModelConfig.max_position_embeddings}")
    
    # Initialize model
    print("\n🔧 Initializing model...")
    model = GPTModel(ModelConfig)
    
    # Parameter count
    total_params, trainable_params = count_parameters(model)
    print(f"\n📊 Model Parameters:")
    print(f"  • Total Parameters: {format_number(total_params)} ({total_params:,})")
    print(f"  • Trainable Parameters: {format_number(trainable_params)} ({trainable_params:,})")
    
    # Memory estimation
    param_size_mb = (total_params * 4) / (1024 ** 2)  # 4 bytes per float32
    print(f"\n💾 Memory Requirements (FP32):")
    print(f"  • Model Size: {param_size_mb:.2f} MB")
    print(f"  • Training (w/ optimizer): ~{param_size_mb * 3:.2f} MB")
    print(f"  • Inference: ~{param_size_mb:.2f} MB")
    
    # Layer breakdown
    print(f"\n🏗️ Architecture Breakdown:")
    print(f"  • Token Embedding: {format_number(ModelConfig.vocab_size * ModelConfig.d_model)} params")
    print(f"  • Position Embedding: {format_number(ModelConfig.max_position_embeddings * ModelConfig.d_model)} params")
    print(f"  • {ModelConfig.n_layers} Transformer Blocks:")
    print(f"    - Each block: Multi-Head Attention + Feed-Forward")
    print(f"    - Attention heads per block: {ModelConfig.n_heads}")
    print(f"    - Head dimension: {ModelConfig.d_model // ModelConfig.n_heads}")
    print(f"  • Language Modeling Head: Tied with token embedding")
    
    # Computational requirements
    seq_len = ModelConfig.max_seq_length
    flops_per_token = 2 * total_params + 2 * ModelConfig.n_layers * ModelConfig.n_heads * seq_len * (ModelConfig.d_model // ModelConfig.n_heads)
    print(f"\n⚡ Computational Requirements:")
    print(f"  • FLOPs per forward pass (seq_len={seq_len}): ~{format_number(flops_per_token * seq_len)}")
    print(f"  • Attention pattern: Causal (autoregressive)")
    print(f"  • Activation function: GELU")
    
    # Comparison with known models
    print(f"\n📏 Comparison with Known Models:")
    if total_params < 125e6:
        print(f"  • Similar to: GPT-2 Small (124M params)")
        print(f"  • Your model: {format_number(total_params)}")
    elif total_params < 350e6:
        print(f"  • Between: GPT-2 Small (124M) and Medium (355M)")
        print(f"  • Your model: {format_number(total_params)}")
    elif total_params < 774e6:
        print(f"  • Similar to: GPT-2 Medium (355M params)")
        print(f"  • Your model: {format_number(total_params)}")
    elif total_params < 1.5e9:
        print(f"  • Similar to: GPT-2 Large (774M params)")
        print(f"  • Your model: {format_number(total_params)}")
    else:
        print(f"  • Similar to: GPT-2 XL (1.5B params) or larger")
        print(f"  • Your model: {format_number(total_params)}")
    
    print("\n" + "="*70)
    print("✅ Model initialized successfully!")
    print("="*70)


def print_layer_details():
    """Print detailed layer-by-layer breakdown"""
    print("\n📦 Detailed Layer-by-Layer Breakdown:")
    print("-" * 70)
    
    model = GPTModel(ModelConfig)
    
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"{name:30s}: {format_number(params):>10s} parameters")
    
    print("-" * 70)


if __name__ == "__main__":
    print_model_info()
    print_layer_details()
    
    print("\n💡 Tips:")
    print("  • To reduce model size: Decrease n_layers, n_heads, or d_model")
    print("  • To increase model size: Increase n_layers, n_heads, or d_model")
    print("  • Current configuration is similar to GPT-2 Small/Medium")
    print("  • Edit configs/config.py to modify architecture")
