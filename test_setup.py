"""
Test script to verify installation and setup
"""

import sys

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  - MPS available: {torch.backends.mps.is_available()}")
        print(f"  - MPS built: {torch.backends.mps.is_built()}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        return False
    
    try:
        import kagglehub
        print(f"✓ Kagglehub")
    except ImportError as e:
        print(f"✗ Kagglehub import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✓ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False
    
    try:
        from tokenizers import Tokenizer
        print(f"✓ Tokenizers")
    except ImportError as e:
        print(f"✗ Tokenizers import failed: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    return True


def test_project_structure():
    """Test project structure"""
    print("\nTesting project structure...")
    
    import os
    
    required_files = [
        "train.py",
        "generate.py",
        "requirements.txt",
        "README.md",
        "QUICKSTART.md",
        "configs/config.py",
        "src/model/gpt.py",
        "src/data/dataset.py",
        "src/data/tokenizer.py",
        "src/data/download_dataset.py",
        "src/training/trainer.py",
        "src/utils/utils.py"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING!")
            all_exist = False
    
    return all_exist


def test_model_import():
    """Test model can be imported"""
    print("\nTesting model import...")
    
    try:
        from src.model.gpt import GPTModel
        from configs.config import ModelConfig
        print("✓ Model imports successful")
        return True
    except Exception as e:
        print(f"✗ Model import failed: {e}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("LLM Project Setup Verification")
    print("="*60 + "\n")
    
    results = []
    
    # Test imports
    results.append(("Imports", test_imports()))
    
    # Test project structure
    results.append(("Project Structure", test_project_structure()))
    
    # Test model import
    results.append(("Model Import", test_model_import()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results:
        status = "PASSED ✓" if passed else "FAILED ✗"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("All tests passed! ✓")
        print("\nYou're ready to start training!")
        print("\nNext steps:")
        print("1. Run: python src/data/download_dataset.py")
        print("2. Run: python train.py")
        print("3. Run: python generate.py --interactive")
    else:
        print("Some tests failed. Please check the errors above.")
    print("="*60)


if __name__ == "__main__":
    main()
