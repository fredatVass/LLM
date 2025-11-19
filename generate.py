"""
Text generation script using trained GPT model
"""

import torch
import sys
import os
from pathlib import Path
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from configs.config import ModelConfig, GenerationConfig
from src.model.gpt import GPTModel
from src.data.tokenizer import SimpleTokenizer


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    print(f"Loading model from {checkpoint_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize model
    model = GPTModel(ModelConfig)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    return model


def generate_text(model, tokenizer, prompt, config, device):
    """Generate text from prompt"""
    print(f"\nPrompt: {prompt}")
    print("-" * 60)
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            max_length=config.max_length,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode generated text
    generated_text = tokenizer.decode(generated_ids[0])
    
    print(f"Generated: {generated_text}")
    print("-" * 60)
    
    return generated_text


def interactive_mode(model, tokenizer, config, device):
    """Interactive text generation"""
    print("\n" + "="*60)
    print("Interactive Text Generation Mode")
    print("="*60)
    print("Enter your prompts below. Type 'quit' to exit.")
    print("Type 'config' to adjust generation parameters.")
    print("-" * 60)
    
    while True:
        try:
            prompt = input("\nPrompt: ").strip()
            
            if prompt.lower() == 'quit':
                print("Exiting...")
                break
            
            if prompt.lower() == 'config':
                print("\nCurrent configuration:")
                print(f"  max_length: {config.max_length}")
                print(f"  temperature: {config.temperature}")
                print(f"  top_k: {config.top_k}")
                print(f"  top_p: {config.top_p}")
                print(f"  repetition_penalty: {config.repetition_penalty}")
                continue
            
            if not prompt:
                print("Please enter a prompt.")
                continue
            
            # Generate text
            generated_text = generate_text(model, tokenizer, prompt, config, device)
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main generation function"""
    parser = argparse.ArgumentParser(description="Generate text using trained GPT model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for generation"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=GenerationConfig.max_length,
        help="Maximum length of generated text"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=GenerationConfig.temperature,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=GenerationConfig.top_k,
        help="Top-k sampling"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=GenerationConfig.top_p,
        help="Top-p (nucleus) sampling"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    args = parser.parse_args()
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = SimpleTokenizer()
    tokenizer.load("tokenizer")
    
    # Load model
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Please train the model first using: python train.py")
        sys.exit(1)
    
    model = load_model(args.checkpoint, device)
    
    # Update generation config
    config = GenerationConfig()
    config.max_length = args.max_length
    config.temperature = args.temperature
    config.top_k = args.top_k
    config.top_p = args.top_p
    
    # Generate text
    if args.interactive:
        interactive_mode(model, tokenizer, config, device)
    elif args.prompt:
        generate_text(model, tokenizer, args.prompt, config, device)
    else:
        # Default prompts
        default_prompts = [
            "Once upon a time",
            "The future of artificial intelligence",
            "In a world where technology",
            "The most important thing about machine learning"
        ]
        
        print("\n" + "="*60)
        print("Generating text with default prompts")
        print("="*60)
        
        for prompt in default_prompts:
            generate_text(model, tokenizer, prompt, config, device)
            print("\n")


if __name__ == "__main__":
    main()
