# GPT Model Training Project - Complete Documentation

## 🎉 Project Successfully Created!

You now have a complete, production-ready GPT language model implementation optimized for your Mac M4.

## 📁 Project Structure

```
LLM-1/
├── configs/
│   ├── config.py              # All configuration settings
│   └── __init__.py
├── src/
│   ├── model/
│   │   ├── gpt.py            # GPT architecture implementation
│   │   └── __init__.py
│   ├── data/
│   │   ├── download_dataset.py  # Dataset downloader
│   │   ├── dataset.py           # PyTorch Dataset class
│   │   ├── tokenizer.py         # BPE tokenizer
│   │   └── __init__.py
│   ├── training/
│   │   ├── trainer.py        # Training loop
│   │   └── __init__.py
│   ├── utils/
│   │   ├── utils.py          # Helper functions
│   │   └── __init__.py
│   └── __init__.py
├── checkpoints/              # Model checkpoints (created during training)
├── data/                     # Dataset storage (created during download)
├── tokenizer/               # Tokenizer files (created during training)
├── train.py                 # Main training script
├── generate.py             # Text generation script
├── test_setup.py           # Setup verification
├── requirements.txt        # Dependencies
├── README.md              # Project overview
├── QUICKSTART.md          # Quick start guide
└── .gitignore            # Git ignore rules
```

## 🚀 Quick Start (3 Simple Steps)

### Step 1: Download Dataset
```bash
python src/data/download_dataset.py
```

This downloads the LLM-7 prompt training dataset from Kaggle and preprocesses it.

**Note**: You'll need Kaggle credentials. If you don't have them:
1. Go to https://www.kaggle.com/settings
2. Click "Create New API Token"
3. Move the downloaded `kaggle.json` to `~/.kaggle/`

### Step 2: Train the Model
```bash
python train.py
```

This will:
- Automatically train a tokenizer (first run)
- Train the GPT model on your data
- Save checkpoints periodically
- Use MPS acceleration on your Mac M4

Training typically takes several hours depending on dataset size and model configuration.

### Step 3: Generate Text
```bash
python generate.py --interactive
```

Or with a specific prompt:
```bash
python generate.py --prompt "Once upon a time"
```

## 🏗️ Architecture Details

### GPT Model Components

1. **Token Embedding Layer**
   - Converts token IDs to dense vectors
   - Dimension: 768 (configurable)

2. **Positional Encoding**
   - Learned positional embeddings
   - Supports sequences up to 1024 tokens

3. **Transformer Blocks (12 layers)**
   - Multi-Head Self-Attention (12 heads)
   - Position-wise Feed-Forward Network
   - Layer Normalization (pre-norm architecture)
   - Residual connections
   - Dropout for regularization

4. **Language Modeling Head**
   - Projects to vocabulary size
   - Weight tying with embedding layer

### Key Features

- **Causal Masking**: Ensures autoregressive generation
- **Scaled Dot-Product Attention**: Efficient attention mechanism
- **GELU Activation**: Modern activation function
- **Gradient Clipping**: Prevents exploding gradients
- **Learning Rate Scheduling**: Cosine annealing or linear decay
- **Gradient Accumulation**: Train with larger effective batch sizes

## ⚙️ Configuration

All settings are in `configs/config.py`:

### Model Size Configuration

**Current (Medium-sized model)**
```python
n_layers = 12      # Number of transformer blocks
n_heads = 12       # Attention heads per block
d_model = 768      # Model dimension
d_ff = 3072       # Feed-forward dimension
```

**Small (for testing)**
```python
n_layers = 6
n_heads = 8
d_model = 512
d_ff = 2048
```

**Large (for better performance)**
```python
n_layers = 24
n_heads = 16
d_model = 1024
d_ff = 4096
```

### Training Configuration

```python
batch_size = 16                    # Adjust for memory
learning_rate = 3e-4               # AdamW learning rate
max_epochs = 10                    # Training epochs
gradient_accumulation_steps = 4    # Effective batch = 16 * 4 = 64
max_grad_norm = 1.0               # Gradient clipping
```

### Generation Configuration

```python
max_length = 200           # Tokens to generate
temperature = 0.8          # Sampling randomness (0.1-2.0)
top_k = 50                # Consider top-k tokens
top_p = 0.95              # Nucleus sampling threshold
repetition_penalty = 1.2  # Discourage repetition
```

## 🎯 Training Features

### Automatic Features

1. **Tokenizer Management**
   - Trains BPE tokenizer on first run
   - Automatically loads on subsequent runs
   - Saves to `tokenizer/` directory

2. **Checkpointing**
   - Saves every 1000 steps
   - Saves best model based on validation loss
   - Saves final model after training
   - All checkpoints in `checkpoints/`

3. **Monitoring**
   - Loss logging every 100 steps
   - Validation every 500 steps
   - Learning rate tracking
   - Progress bars with tqdm

4. **Device Optimization**
   - Automatic MPS detection (Mac M4)
   - Falls back to CUDA or CPU
   - Memory-efficient implementation

### Training Outputs

```
checkpoints/
├── best_model.pt              # Best validation loss
├── final_model.pt             # End of training
└── checkpoint_step_XXXX.pt    # Periodic checkpoints
```

## 📊 Monitoring Training

### View Real-time Logs
```bash
tail -f training.log
```

### Expected Output
```
Step 100, Epoch 0, Loss: 4.5234, LR: 0.000295
Step 200, Epoch 0, Loss: 4.1234, LR: 0.000290
Validation Loss: 3.9876
New best model saved! Val Loss: 3.9876
```

### Loss Expectations

- **Initial Loss**: 8-10 (random predictions)
- **After 1000 steps**: 4-6
- **After 5000 steps**: 2-4
- **Well-trained**: 1-2.5

Lower is better! Perplexity = exp(loss)

## 🎨 Text Generation

### Basic Usage

```bash
# Interactive mode (recommended for exploration)
python generate.py --interactive

# Single prompt
python generate.py --prompt "The future of AI"

# Default demo with multiple prompts
python generate.py
```

### Advanced Generation

```bash
python generate.py \
  --prompt "In a world where" \
  --max_length 300 \
  --temperature 0.9 \
  --top_k 40 \
  --top_p 0.92 \
  --checkpoint checkpoints/best_model.pt
```

### Generation Parameters Explained

- **temperature**: Controls randomness
  - Low (0.1-0.5): More focused, deterministic
  - Medium (0.6-0.9): Balanced
  - High (1.0-2.0): More creative, diverse

- **top_k**: Limits vocabulary at each step
  - Lower (10-30): More focused
  - Higher (50-100): More diverse

- **top_p** (nucleus sampling): Cumulative probability
  - 0.9: Very focused
  - 0.95: Balanced
  - 0.99: More diverse

- **repetition_penalty**: Discourages repetition
  - 1.0: No penalty
  - 1.2: Moderate penalty (recommended)
  - 1.5+: Strong penalty

## 🔧 Troubleshooting

### Out of Memory

**Symptoms**: Process killed, "RuntimeError: out of memory"

**Solutions**:
1. Reduce batch size: `batch_size = 8` or `4`
2. Reduce sequence length: `max_seq_length = 256`
3. Increase gradient accumulation: `gradient_accumulation_steps = 8`
4. Reduce model size (fewer layers/smaller dimensions)

### Slow Training

**Symptoms**: < 1 step per second

**Check**:
1. Verify MPS is being used (should see "Using MPS" in output)
2. Reduce model size or batch size
3. Close other applications

### Poor Generation Quality

**Symptoms**: Nonsensical or repetitive text

**Solutions**:
1. Train longer (more epochs/steps)
2. Check training loss (should be < 3.0)
3. Adjust generation parameters
4. Increase model size
5. Ensure dataset quality

### Kaggle Dataset Download Fails

**Symptoms**: "Permission denied" or "401 Unauthorized"

**Solutions**:
1. Set up Kaggle credentials:
   ```bash
   mkdir -p ~/.kaggle
   # Copy your kaggle.json here
   chmod 600 ~/.kaggle/kaggle.json
   ```
2. Verify credentials at https://www.kaggle.com/settings
3. Check internet connection

## 🎓 Understanding the Training Process

### What Happens During Training?

1. **Initialization**
   - Load and preprocess dataset
   - Train/load tokenizer
   - Initialize model with random weights
   - Set up optimizer and scheduler

2. **Training Loop** (for each batch)
   - Forward pass: Predict next tokens
   - Calculate loss: Compare predictions to actual
   - Backward pass: Compute gradients
   - Optimizer step: Update weights
   - Scheduler step: Adjust learning rate

3. **Validation**
   - Periodically evaluate on held-out data
   - No weight updates
   - Monitor for overfitting

4. **Checkpointing**
   - Save model state periodically
   - Keep best performing model
   - Enable resume from failure

### Loss Function

The model uses **Cross-Entropy Loss**:
- Measures how well model predicts next token
- Lower loss = better predictions
- Target: < 2.5 for decent text generation

### Optimization

- **Optimizer**: AdamW (Adam with weight decay)
- **Learning Rate**: Starts at 3e-4
- **Schedule**: Cosine annealing (gradually decreases)
- **Gradient Clipping**: Prevents training instability

## 🚀 Advanced Usage

### Resume Training

Modify `train.py`:
```python
# After creating trainer
trainer.load_checkpoint("checkpoints/checkpoint_step_5000.pt")
trainer.train()
```

### Fine-tune on Custom Data

1. Prepare your text data (one document per line)
2. Save as `data/processed/train.txt`
3. Run training normally

### Distributed Training (Future)

Currently single-device. For multi-GPU:
- Implement `DistributedDataParallel`
- Add multi-GPU configuration
- Adjust batch size accordingly

### Export for Production

```python
import torch
from src.model.gpt import GPTModel
from configs.config import ModelConfig

# Load model
model = GPTModel(ModelConfig)
checkpoint = torch.load("checkpoints/best_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])

# Export just the model weights
torch.save(model.state_dict(), "model_production.pt")
```

### Integration with Applications

```python
# In your application
import torch
from src.model.gpt import GPTModel
from src.data.tokenizer import SimpleTokenizer

# Load model and tokenizer
model = GPTModel(config)
model.load_state_dict(torch.load("model_production.pt"))
tokenizer = SimpleTokenizer()
tokenizer.load("tokenizer")

# Generate text
def generate(prompt):
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids])
    output = model.generate(input_ids, max_length=100)
    return tokenizer.decode(output[0])
```

## 📈 Performance Optimization

### Mac M4 Specific

1. **MPS Acceleration**: Automatically enabled
2. **Unified Memory**: Efficient memory usage
3. **Neural Engine**: May accelerate some ops

### General Tips

1. **Batch Size**: Maximize without OOM
2. **Workers**: Set `num_workers = 4` in DataLoader
3. **Pin Memory**: Already enabled
4. **Mixed Precision**: Future enhancement

## 🔬 Model Evaluation

### Metrics to Track

1. **Training Loss**: Should decrease steadily
2. **Validation Loss**: Should decrease, not diverge from train
3. **Perplexity**: exp(loss), lower is better
4. **Generation Quality**: Manual inspection

### Evaluation Script (Future Enhancement)

```python
# Calculate perplexity on test set
# BLEU scores for specific tasks
# Human evaluation ratings
```

## 🎯 Next Steps

### Immediate
1. ✅ Download dataset
2. ✅ Train model
3. ✅ Generate text
4. Experiment with parameters

### Short-term
1. Fine-tune on specific domain
2. Implement better evaluation
3. Add more generation strategies
4. Create web interface

### Long-term
1. Scale up model size
2. Implement instruction tuning
3. Add RLHF (Reinforcement Learning from Human Feedback)
4. Multi-modal capabilities
5. Deploy as API service

## 📚 Learning Resources

### Understanding Transformers
- "Attention Is All You Need" paper
- The Illustrated Transformer (Jay Alammar)
- Stanford CS224N (NLP with Deep Learning)

### PyTorch
- Official PyTorch tutorials
- PyTorch documentation

### LLMs
- Andrej Karpathy's lectures
- Hugging Face course
- OpenAI papers (GPT series)

## 🤝 Contributing

To extend this project:

1. **Add features** in appropriate modules
2. **Update configs** as needed
3. **Document changes** in code
4. **Test thoroughly** before deployment

## 📄 License

This is an educational project. Adapt and use as needed for your purposes.

## 🎊 Congratulations!

You now have a complete, working GPT model implementation! 

**What you've built**:
- ✅ Custom transformer architecture
- ✅ BPE tokenizer
- ✅ Complete training pipeline
- ✅ Text generation system
- ✅ Monitoring and checkpointing
- ✅ Mac M4 optimizations

**Ready to train your LLM!** 🚀

---

*Created: November 19, 2025*
*For: Mac M4 with MPS acceleration*
*Model: GPT-style decoder-only transformer*
