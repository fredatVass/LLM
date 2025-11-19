#!/bin/bash
# Training Monitor Script

echo "======================================================"
echo "🚀 GPT Model Training Monitor"
echo "======================================================"
echo ""

# Check if training is running
if ps aux | grep "train.py" | grep -v grep > /dev/null; then
    echo "✅ Training Status: RUNNING"
    PID=$(ps aux | grep "train.py" | grep -v grep | awk '{print $2}')
    echo "   Process ID: $PID"
else
    echo "❌ Training Status: NOT RUNNING"
    echo "   Start training with: python train.py"
fi

echo ""
echo "======================================================"
echo "📊 Latest Training Progress"
echo "======================================================"

if [ -f "training_output.log" ]; then
    echo ""
    # Show last 20 lines with Step information
    grep -E "Step|Loss|Epoch|Validation|best|checkpoint" training_output.log | tail -20
else
    echo "No training log found yet."
fi

echo ""
echo "======================================================"
echo "📁 Checkpoints"
echo "======================================================"

if [ -d "checkpoints" ]; then
    ls -lh checkpoints/*.pt 2>/dev/null | awk '{print $9, "-", $5}' || echo "No checkpoints yet."
else
    echo "Checkpoints directory not created yet."
fi

echo ""
echo "======================================================"
echo "💡 Useful Commands"
echo "======================================================"
echo "  Monitor live:  tail -f training_output.log"
echo "  Stop training: kill -9 $PID" 
echo "  View logs:     less training_output.log"
echo "  Run monitor:   ./monitor_training.sh"
echo "======================================================"
