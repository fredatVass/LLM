# GPT Language Model

A GPT-based language model implementation with interpretable white box capabilities.

## Features

- Custom GPT architecture (124M parameters)
- Training pipeline optimized for Mac M4 (MPS support)
- BPE tokenization
- White box interpretability (explain model decisions)
- Black box inference (standard predictions)
- Text generation
- Attention visualization

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download dataset:
```bash
python src/data/download_dataset.py
```

3. Train model:
```bash
python train.py
```

4. Generate text:
```bash
python generate.py --prompt "Your text here"
```

5. Use interpretable model (after training):
```bash
python demo_interpretable.py
```

## Project Structure

```
src/
├── model/              # GPT architecture and interpretable wrapper
├── data/               # Data loading and tokenization
├── training/           # Training pipeline
└── utils/              # Helper functions
configs/                # Model and training configuration
train.py               # Main training script
generate.py            # Text generation
demo_interpretable.py  # Interpretable model demo
```

## Configuration

Edit `configs/config.py` to adjust model size, training parameters, and generation settings.

## White Box vs Black Box

The model supports two modes:
- **Black Box**: Standard predictions (fast)
- **White Box**: Full explanations of decisions (transparent)

See `demo_interpretable.py` for examples.
