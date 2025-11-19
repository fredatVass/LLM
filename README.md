# LLM Training Project

A comprehensive GPT-based Language Model implementation with training pipeline for text generation.

## Features

- Custom GPT architecture with multi-head self-attention
- Efficient training pipeline optimized for Mac M4 (MPS support)
- Tokenization with BPE (Byte Pair Encoding)
- Automatic dataset download from Kaggle
- Checkpointing and model saving
- Text generation inference
- Comprehensive logging and evaluation metrics

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download and prepare dataset:
```bash
python src/data/download_dataset.py
```

3. Train the model:
```bash
python train.py
```

4. Generate text:
```bash
python generate.py --prompt "Your prompt here"
```

## Project Structure

```
.
├── src/
│   ├── model/          # GPT model architecture
│   ├── data/           # Data loading and preprocessing
│   ├── training/       # Training loop and utilities
│   └── utils/          # Helper functions
├── configs/            # Configuration files
├── checkpoints/        # Model checkpoints
├── data/              # Dataset storage
├── train.py           # Main training script
├── generate.py        # Text generation script
└── requirements.txt   # Dependencies
```

## Configuration

Edit `configs/config.py` to adjust:
- Model size (layers, heads, dimensions)
- Training hyperparameters
- Dataset parameters
- Generation settings

## Hardware

Optimized for Apple Silicon (M1/M2/M3/M4) with MPS backend support.
