#!/usr/bin/env python3
"""
Real-time training monitor with automatic plot updates
"""

import time
import subprocess
import os
import sys

def check_training_status():
    """Check if training is running"""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        return 'train.py' in result.stdout
    except:
        return False

def get_latest_progress():
    """Get latest training progress"""
    try:
        result = subprocess.run(
            ['grep', 'Step', 'training_output.log'],
            capture_output=True,
            text=True,
            cwd='/Users/fred/LLM-1'
        )
        lines = result.stdout.strip().split('\n')
        if lines and lines[0]:
            return len(lines), lines[-1]
        return 0, None
    except:
        return 0, None

def monitor_and_plot(interval=60, max_iterations=120):
    """Monitor training and update plots periodically"""
    
    print("="*70)
    print("🔄 Real-Time Training Monitor")
    print("="*70)
    print(f"Monitoring interval: {interval} seconds")
    print(f"Will check for up to {max_iterations} iterations ({max_iterations * interval / 3600:.1f} hours)")
    print("\nPress Ctrl+C to stop monitoring\n")
    
    last_step_count = 0
    iteration = 0
    
    try:
        while iteration < max_iterations:
            iteration += 1
            
            # Check if training is still running
            is_running = check_training_status()
            
            # Get current progress
            step_count, latest_line = get_latest_progress()
            
            # Clear screen (optional)
            # os.system('clear')
            
            print(f"\n[Iteration {iteration}] {time.strftime('%H:%M:%S')}")
            print("-" * 70)
            
            if is_running:
                print("Status: ✅ Training RUNNING")
            else:
                print("Status: ⚠️  Training NOT RUNNING (may have finished)")
            
            if latest_line:
                print(f"Latest: {latest_line}")
                print(f"Total logged steps: {step_count}")
                
                # Check if new steps have been logged
                if step_count > last_step_count:
                    new_steps = step_count - last_step_count
                    print(f"New steps since last check: {new_steps}")
                    last_step_count = step_count
                    
                    # Update plots if we have enough data
                    if step_count >= 5:
                        print("\n📊 Updating plots...")
                        subprocess.run(['/usr/bin/python3', 'plot_curves.py'], 
                                     cwd='/Users/fred/LLM-1',
                                     capture_output=True)
                        print("✅ Plots updated: training_curves.png")
                else:
                    print("⏳ No new steps logged yet...")
            else:
                print("⏳ Waiting for training to start logging...")
            
            # Check for checkpoints
            checkpoint_dir = '/Users/fred/LLM-1/checkpoints'
            if os.path.exists(checkpoint_dir):
                checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
                if checkpoints:
                    print(f"\n💾 Checkpoints saved: {len(checkpoints)}")
                    for cp in sorted(checkpoints)[-3:]:  # Show last 3
                        print(f"   - {cp}")
            
            print("-" * 70)
            
            # Stop if training finished
            if not is_running and step_count > 0:
                print("\n✅ Training appears to have finished!")
                print("\nGenerating final plots...")
                subprocess.run(['/usr/bin/python3', 'plot_curves.py'], 
                             cwd='/Users/fred/LLM-1')
                break
            
            # Wait before next check
            if iteration < max_iterations:
                print(f"\n⏰ Next check in {interval} seconds... (Ctrl+C to stop)")
                time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Monitoring stopped by user")
        print("\nGenerating current plots...")
        subprocess.run(['/usr/bin/python3', 'plot_curves.py'], 
                     cwd='/Users/fred/LLM-1')
    
    print("\n" + "="*70)
    print("Monitoring session ended")
    print("="*70)
    print("\nUseful commands:")
    print("  - View plots: open training_curves.png")
    print("  - Check logs: tail -f training_output.log")
    print("  - Manual plot: python plot_curves.py")
    print("  - Monitor again: python monitor_live.py")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor GPT training in real-time')
    parser.add_argument('--interval', type=int, default=120,
                       help='Check interval in seconds (default: 120)')
    parser.add_argument('--duration', type=int, default=240,
                       help='Total monitoring duration in minutes (default: 240 = 4 hours)')
    
    args = parser.parse_args()
    
    max_iterations = (args.duration * 60) // args.interval
    
    monitor_and_plot(interval=args.interval, max_iterations=max_iterations)
