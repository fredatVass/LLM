"""
Configuration file for the GPT model and training pipeline.
"""

class ModelConfig:
    """GPT Model Architecture Configuration"""
    # Model architecture
    vocab_size = 50000  # Will be set after tokenizer training
    max_seq_length = 512  # Maximum sequence length
    n_layers = 12  # Number of transformer layers
    n_heads = 12  # Number of attention heads
    d_model = 768  # Model dimension
    d_ff = 3072  # Feedforward dimension (typically 4 * d_model)
    dropout = 0.1  # Dropout rate
    
    # Positional encoding
    max_position_embeddings = 1024
    
    # Layer normalization
    layer_norm_epsilon = 1e-5


class TrainingConfig:
    """Training Configuration"""
    # Training hyperparameters
    batch_size = 16  # Adjust based on M4 memory
    learning_rate = 3e-4
    weight_decay = 0.01
    max_epochs = 10
    warmup_steps = 500
    max_steps = 100000
    
    # Gradient settings
    gradient_accumulation_steps = 4
    max_grad_norm = 1.0
    
    # Learning rate schedule
    lr_scheduler = "cosine"  # Options: "cosine", "linear"
    
    # Device
    device = "mps"  # Mac M4 GPU acceleration
    
    # Checkpointing
    save_every_n_steps = 1000
    eval_every_n_steps = 500
    checkpoint_dir = "checkpoints"
    
    # Logging
    log_every_n_steps = 100
    wandb_logging = False  # Set to True if using Weights & Biases


class DataConfig:
    """Data Processing Configuration"""
    # Dataset paths
    dataset_name = "carlmcbrideellis/llm-7-prompt-training-dataset"
    data_dir = "data"
    processed_data_dir = "data/processed"
    
    # Tokenizer
    tokenizer_dir = "tokenizer"
    vocab_size = 50000
    
    # Data processing
    train_split = 0.9
    val_split = 0.05
    test_split = 0.05
    
    # Data loading
    num_workers = 4
    shuffle = True


class GenerationConfig:
    """Text Generation Configuration"""
    max_length = 200
    temperature = 0.8
    top_k = 50
    top_p = 0.95
    num_return_sequences = 1
    repetition_penalty = 1.2
    do_sample = True
