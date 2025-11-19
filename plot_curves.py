"""
Plot training and validation learning curves
"""

import matplotlib.pyplot as plt
import re
import os
import sys
from datetime import datetime

def parse_training_log(log_file="training_output.log"):
    """Parse training log to extract loss values"""
    
    if not os.path.exists(log_file):
        print(f"Error: {log_file} not found!")
        print("Make sure training has started and is generating logs.")
        return None, None
    
    train_steps = []
    train_losses = []
    val_steps = []
    val_losses = []
    learning_rates = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # Parse training step: "Step 100, Epoch 0, Loss: 4.5234, LR: 0.000295"
            train_match = re.search(r'Step (\d+), Epoch \d+, Loss: ([\d.]+), LR: ([\d.]+)', line)
            if train_match:
                step = int(train_match.group(1))
                loss = float(train_match.group(2))
                lr = float(train_match.group(3))
                train_steps.append(step)
                train_losses.append(loss)
                learning_rates.append(lr)
            
            # Parse validation: "Validation Loss: 3.9876"
            val_match = re.search(r'Validation at step (\d+): Loss=([\d.]+)', line)
            if not val_match:
                # Alternative format: "Validation Loss: 3.9876"
                if "Validation Loss:" in line:
                    val_match = re.search(r'Validation Loss: ([\d.]+)', line)
                    if val_match and train_steps:
                        val_loss = float(val_match.group(1))
                        val_steps.append(train_steps[-1])  # Use last training step
                        val_losses.append(val_loss)
            else:
                step = int(val_match.group(1))
                val_loss = float(val_match.group(2))
                val_steps.append(step)
                val_losses.append(val_loss)
    
    return {
        'train_steps': train_steps,
        'train_losses': train_losses,
        'val_steps': val_steps,
        'val_losses': val_losses,
        'learning_rates': learning_rates
    }


def plot_learning_curves(data, save_path="training_curves.png"):
    """Plot training and validation curves"""
    
    if data is None:
        return
    
    train_steps = data['train_steps']
    train_losses = data['train_losses']
    val_steps = data['val_steps']
    val_losses = data['val_losses']
    learning_rates = data['learning_rates']
    
    if not train_steps:
        print("No training data found in log file yet.")
        print("Training may just be starting. Wait a few minutes and try again.")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Loss curves
    ax1 = axes[0]
    
    # Plot training loss
    if train_steps:
        ax1.plot(train_steps, train_losses, 'b-', alpha=0.6, linewidth=1, label='Training Loss')
        # Add smoothed line
        if len(train_losses) > 10:
            window_size = min(50, len(train_losses) // 10)
            smoothed = []
            for i in range(len(train_losses)):
                start = max(0, i - window_size)
                end = min(len(train_losses), i + window_size)
                smoothed.append(sum(train_losses[start:end]) / (end - start))
            ax1.plot(train_steps, smoothed, 'b-', linewidth=2, label='Training Loss (smoothed)')
    
    # Plot validation loss
    if val_steps:
        ax1.plot(val_steps, val_losses, 'ro-', linewidth=2, markersize=6, label='Validation Loss')
    
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add text with current stats
    if train_losses:
        current_loss = train_losses[-1]
        min_loss = min(train_losses)
        ax1.text(0.02, 0.98, f'Current Loss: {current_loss:.4f}\nMin Train Loss: {min_loss:.4f}',
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)
    
    if val_losses:
        best_val_loss = min(val_losses)
        best_val_step = val_steps[val_losses.index(best_val_loss)]
        ax1.text(0.98, 0.98, f'Best Val Loss: {best_val_loss:.4f}\n@ Step {best_val_step}',
                transform=ax1.transAxes, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
                fontsize=10)
    
    # Plot 2: Learning rate schedule
    ax2 = axes[1]
    
    if learning_rates:
        ax2.plot(train_steps, learning_rates, 'g-', linewidth=2)
        ax2.set_xlabel('Training Steps', fontsize=12)
        ax2.set_ylabel('Learning Rate', fontsize=12)
        ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add current LR text
        current_lr = learning_rates[-1]
        ax2.text(0.02, 0.98, f'Current LR: {current_lr:.6f}',
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
                fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Learning curves saved to: {save_path}")
    
    # Show figure
    plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("📊 Training Statistics")
    print("="*60)
    
    if train_steps:
        print(f"Total Steps: {train_steps[-1]}")
        print(f"Training Samples: {len(train_steps)}")
        print(f"\nTraining Loss:")
        print(f"  Current: {train_losses[-1]:.4f}")
        print(f"  Initial: {train_losses[0]:.4f}")
        print(f"  Best: {min(train_losses):.4f}")
        print(f"  Improvement: {train_losses[0] - train_losses[-1]:.4f}")
        
        # Calculate perplexity
        import math
        current_perplexity = math.exp(train_losses[-1])
        print(f"\nCurrent Perplexity: {current_perplexity:.2f}")
    
    if val_losses:
        print(f"\nValidation Loss:")
        print(f"  Current: {val_losses[-1]:.4f}")
        print(f"  Best: {min(val_losses):.4f}")
        print(f"  Best at Step: {val_steps[val_losses.index(min(val_losses))]}")
        
        best_val_perplexity = math.exp(min(val_losses))
        print(f"  Best Perplexity: {best_val_perplexity:.2f}")
    
    if learning_rates:
        print(f"\nLearning Rate:")
        print(f"  Current: {learning_rates[-1]:.6f}")
        print(f"  Initial: {learning_rates[0]:.6f}")
    
    print("="*60)


def plot_comparison(data, save_path="loss_comparison.png"):
    """Create a focused comparison of train vs validation loss"""
    
    if data is None or not data['val_steps']:
        return
    
    val_steps = data['val_steps']
    val_losses = data['val_losses']
    train_steps = data['train_steps']
    train_losses = data['train_losses']
    
    plt.figure(figsize=(10, 6))
    
    # Plot both on same axis for comparison
    if train_steps:
        plt.plot(train_steps, train_losses, 'b-', alpha=0.3, linewidth=1, label='Training Loss (all steps)')
    
    if val_steps:
        plt.plot(val_steps, val_losses, 'ro-', linewidth=2, markersize=8, label='Validation Loss', zorder=5)
    
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Check for overfitting
    if len(val_losses) > 1:
        if val_losses[-1] > val_losses[-2]:
            plt.text(0.5, 0.95, '⚠️ Validation loss increasing - potential overfitting',
                    transform=plt.gca().transAxes, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                    fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Comparison plot saved to: {save_path}")


def main():
    """Main function"""
    print("="*60)
    print("📈 GPT Training Visualization")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check if training is running
    import subprocess
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        if 'train.py' in result.stdout:
            print("✅ Training is currently running")
        else:
            print("ℹ️  Training process not detected")
    except:
        pass
    
    print("\nParsing training log...")
    data = parse_training_log()
    
    if data and data['train_steps']:
        print(f"Found {len(data['train_steps'])} training steps")
        print(f"Found {len(data['val_steps'])} validation points")
        
        print("\nGenerating plots...")
        plot_learning_curves(data)
        
        if data['val_steps']:
            plot_comparison(data)
        
        print("\n✅ Visualization complete!")
        print("\nTo update the plots, run this script again:")
        print("  python plot_curves.py")
    else:
        print("\n⚠️  No training data found yet.")
        print("\nPossible reasons:")
        print("  1. Training just started (wait a few minutes)")
        print("  2. Log file doesn't exist yet")
        print("  3. Training hasn't reached first logging step")
        print("\nTry again in a few minutes!")


if __name__ == "__main__":
    main()
