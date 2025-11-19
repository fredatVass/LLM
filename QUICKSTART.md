# Quick Start Guide

## Step-by-Step Instructions

### 1. Install Dependencies

First, install all required Python packages:

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch (with MPS support for Mac M4)
- Transformers library
- Kagglehub for dataset downloading
- All other dependencies

### 2. Download and Prepare Dataset

Download the LLM-7 prompt training dataset and preprocess it:

```bash
python src/data/download_dataset.py
```

This will:
- Download the dataset from Kaggle (you may need to configure Kaggle credentials)
- Preprocess the data
- Split into train/validation/test sets
- Save processed data to `data/processed/`

**Note**: If you don't have Kaggle credentials set up, you'll need to:
1. Create a Kaggle account at https://www.kaggle.com
2. Go to Account settings -> API -> Create New API Token
3. This will download `kaggle.json`
4. Place it in `~/.kaggle/kaggle.json` (or follow the instructions shown)

### 3. Train the Model

Start training the GPT model:

```bash
python train.py
```

Training features:
- Automatic tokenizer training (first run) or loading (subsequent runs)
- Progress logging every 100 steps
- Validation every 500 steps
- Automatic checkpointing every 1000 steps
- Best model saving based on validation loss
- Optimized for Mac M4 with MPS acceleration

Training will create:
- `tokenizer/` - Trained tokenizer files
- `checkpoints/` - Model checkpoints
  - `best_model.pt` - Best performing model
  - `final_model.pt` - Final model after training
  - `checkpoint_step_*.pt` - Intermediate checkpoints

### 4. Generate Text

Once training is complete, generate text using your trained model:

**Single prompt generation:**
```bash
python generate.py --prompt "Once upon a time in a distant galaxy"
```

**Interactive mode:**
```bash
python generate.py --interactive
```

**With custom parameters:**
```bash
python generate.py \
  --prompt "The future of AI is" \
  --max_length 200 \
  --temperature 0.8 \
  --top_k 50 \
  --top_p 0.95
```

**Default demo (multiple prompts):**
```bash
python generate.py
```

## Configuration

### Model Architecture

Edit `configs/config.py` to adjust model size:

```python
class ModelConfig:
    n_layers = 12          # Number of transformer layers
    n_heads = 12           # Number of attention heads
    d_model = 768          # Model dimension
    d_ff = 3072           # Feedforward dimension
    max_seq_length = 512  # Maximum sequence length
```

### Training Parameters

Adjust training hyperparameters:

```python
class TrainingConfig:
    batch_size = 16              # Reduce if out of memory
    learning_rate = 3e-4
    max_epochs = 10
    gradient_accumulation_steps = 4
```

### Generation Settings

Customize text generation:

```python
class GenerationConfig:
    max_length = 200           # Maximum generated tokens
    temperature = 0.8          # Higher = more random
    top_k = 50                # Top-k sampling
    top_p = 0.95              # Nucleus sampling
    repetition_penalty = 1.2  # Penalize repetition
```

## Monitoring Training

Watch training progress:

```bash
tail -f training.log
```

## Troubleshooting

### Out of Memory Error
- Reduce `batch_size` in `configs/config.py`
- Reduce `max_seq_length`
- Increase `gradient_accumulation_steps`

### Slow Training
- Ensure MPS is being used (should see "Using MPS" in logs)
- Reduce model size (fewer layers/heads)
- Use smaller batch size with more gradient accumulation

### Dataset Download Issues
- Ensure Kaggle credentials are configured
- Check internet connection
- Try manual download from Kaggle website

## Model Architecture Details

This implementation includes:

1. **Multi-Head Self-Attention**: Parallel attention mechanisms for capturing different aspects
2. **Position-wise Feed-Forward Networks**: Two-layer FFN with GELU activation
3. **Layer Normalization**: Pre-norm architecture for stable training
4. **Positional Encoding**: Learned positional embeddings
5. **Causal Masking**: For autoregressive text generation
6. **Weight Tying**: Shared weights between embedding and output layers

## Advanced Usage

### Resume Training from Checkpoint

Modify `train.py` to load a checkpoint:

```python
trainer = Trainer(...)
trainer.load_checkpoint("checkpoints/checkpoint_step_5000.pt")
trainer.train()
```

### Fine-tune on Custom Data

1. Prepare your text data in the same format
2. Place in `data/processed/train.txt`
3. Run training as normal

### Export Model for Production

```python
import torch
model = GPTModel(ModelConfig)
checkpoint = torch.load("checkpoints/best_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])
torch.save(model.state_dict(), "model_production.pt")
```

## Performance Tips for Mac M4

1. **Use MPS**: Automatic with the provided configuration
2. **Batch Size**: Start with 16, adjust based on memory
3. **Mixed Precision**: MPS automatically optimizes for M4
4. **Memory Management**: Close other applications during training

## Next Steps

After training your base model, you can:

1. **Fine-tune** on specific tasks (Q&A, summarization, etc.)
2. **Add RLHF** (Reinforcement Learning from Human Feedback)
3. **Implement retrieval** for RAG (Retrieval-Augmented Generation)
4. **Scale up** model size for better performance
5. **Deploy** as an API service

Happy training! 🚀
