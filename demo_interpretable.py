"""
Demo script for Interpretable White Box GPT Model
Shows how the same model can work as both black box and white box
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.model.interpretable_gpt import create_interpretable_model
import json


def demo_black_box_mode(model):
    """Demonstrate black box inference (what users typically see)"""
    print("\n" + "="*70)
    print("BLACK BOX MODE - Standard Inference")
    print("="*70)
    print("This is how the model appears to end users:")
    print("Simple input -> prediction output, no explanations\n")
    
    test_inputs = [
        "The future of artificial intelligence",
        "Once upon a time",
        "In the year 2050"
    ]
    
    for text in test_inputs:
        result = model.predict(text, explain=False)
        print(f"Input: '{text}'")
        print(f"  Predicted next token: '{result['black_box_output']['predicted_token']}'")
        print(f"  Confidence: {result['black_box_output']['confidence']:.2%}")
        print()


def demo_white_box_mode(model):
    """Demonstrate white box transparency"""
    print("\n" + "="*70)
    print("WHITE BOX MODE - Full Transparency")
    print("="*70)
    print("This reveals the internal workings and decision process:\n")
    
    text = "The future of artificial intelligence"
    result = model.predict(text, explain=True)
    
    print(f"Input: '{text}'\n")
    
    # Show white box information
    white_box = result['white_box_explanation']
    
    print("Internal Analysis:")
    print(f"  - Model used {white_box['model_internals']['num_layers']} layers")
    print(f"  - {white_box['model_internals']['attention_heads']} attention heads per layer")
    print(f"  - Total parameters: {white_box['model_internals']['total_parameters']:,}")
    
    print(f"\nDecision Quality:")
    confidence = white_box['confidence_breakdown']
    print(f"  - {confidence['decision_quality']}")
    print(f"  - Top prediction confidence: {confidence['top_1_confidence']:.2%}")
    print(f"  - Confidence gap to 2nd choice: {confidence['confidence_gap']:.4f}")
    
    print(f"\nUncertainty Analysis:")
    factors = white_box['decision_factors']
    print(f"  - Entropy: {factors['entropy']:.4f}")
    print(f"  - Uncertainty level: {factors['uncertainty']}")
    
    print(f"\nTop 3 Alternative Predictions:")
    for alt in confidence['alternative_predictions'][:3]:
        print(f"  {alt['rank']}. '{alt['token']}' ({alt['probability']:.2%})")


def demo_comparison(model):
    """Show side-by-side comparison"""
    print("\n" + "="*70)
    print("BLACK BOX vs WHITE BOX COMPARISON")
    print("="*70)
    
    text = "In the year 2050"
    comparison = model.compare_black_vs_white(text)
    
    print(f"\nInput: '{text}'\n")
    
    print("BLACK BOX VIEW (User Perspective):")
    print(f"  {comparison['black_box_view']['description']}")
    print(f"  Output: '{comparison['black_box_view']['output']}'")
    print(f"  Confidence: {comparison['black_box_view']['confidence']:.2%}")
    print(f"  Transparency: {comparison['black_box_view']['transparency']}")
    
    print("\nWHITE BOX VIEW (Developer Perspective):")
    print(f"  {comparison['white_box_view']['description']}")
    print(f"  Output: '{comparison['white_box_view']['output']}'")
    print(f"  Confidence: {comparison['white_box_view']['confidence']:.2%}")
    print(f"  Transparency: {comparison['white_box_view']['transparency']}")
    
    print(f"\nKey Insight: {comparison['key_insight']}")


def demo_step_by_step(model):
    """Show detailed step-by-step explanation"""
    print("\n" + "="*70)
    print("STEP-BY-STEP EXPLANATION")
    print("="*70)
    print("Deep dive into how the model makes a decision:\n")
    
    model.explain_prediction_step_by_step("The quick brown fox")


def demo_generation_with_explanation(model):
    """Generate text with periodic explanations"""
    print("\n" + "="*70)
    print("TEXT GENERATION WITH EXPLANATIONS")
    print("="*70)
    print("Generate text and explain decisions periodically:\n")
    
    result = model.generate_with_explanation(
        prompt="Once upon a time",
        max_length=30,
        explain_every=10
    )
    
    print(f"Prompt: '{result['prompt']}'")
    print(f"Generated: '{result['generated_text']}'")
    print(f"\nGeneration Summary:")
    print(f"  - Tokens generated: {result['num_tokens_generated']}")
    print(f"  - Average confidence: {result['generation_summary']['average_confidence']:.2%}")
    print(f"  - Explained {result['generation_summary']['num_explained_steps']} decision points")


def main():
    """Main demo function"""
    print("="*70)
    print("INTERPRETABLE GPT MODEL DEMO")
    print("White Box Transparency + Black Box Functionality")
    print("="*70)
    
    # Note: This demo requires a trained model
    print("\nNOTE: This demo requires a trained model checkpoint.")
    print("To run the full demo:")
    print("1. Train your model: python train.py")
    print("2. Wait for checkpoints to be saved")
    print("3. Run: python demo_interpretable.py --checkpoint checkpoints/best_model.pt")
    print("\n" + "="*70)
    
    # Check if model exists
    checkpoint_path = "checkpoints/best_model.pt"
    tokenizer_path = "tokenizer"
    
    import os
    if not os.path.exists(checkpoint_path):
        print("\nNo trained model found.")
        print(f"   Expected: {checkpoint_path}")
        print("\nTo create a trained model:")
        print("   python train.py")
        print("\nExiting demo...")
        return
    
    print("\nFound trained model! Loading...\n")
    
    # Create interpretable model
    model = create_interpretable_model(
        model_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        device='mps'
    )
    
    # Run demos
    print("\n" + "="*70)
    print("Starting Demonstrations...")
    print("="*70)
    
    # 1. Black box mode
    demo_black_box_mode(model)
    
    # 2. White box mode
    demo_white_box_mode(model)
    
    # 3. Comparison
    demo_comparison(model)
    
    # 4. Step-by-step
    demo_step_by_step(model)
    
    # 5. Generation with explanation
    demo_generation_with_explanation(model)
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\nKey Takeaways:")
    print("  - Same model works as both black box and white box")
    print("  - Black box: Simple predictions for end users")
    print("  - White box: Full transparency for developers/researchers")
    print("  - Can switch between modes on demand")
    print("  - Provides interpretability without sacrificing functionality")
    print("\nUse Cases:")
    print("  - End users: Black box mode (simple, fast)")
    print("  - Debugging: White box mode (understand failures)")
    print("  - Research: Full transparency into model behavior")
    print("  - Auditing: Verify model decisions and confidence")
    print("  - Education: Learn how LLMs make decisions")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Demo interpretable GPT model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str, default='tokenizer',
                       help='Path to tokenizer directory')
    
    args = parser.parse_args()
    
    main()
