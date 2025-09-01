#!/usr/bin/env python3
"""
Verify project setup and structure
"""

import os
import sys
from pathlib import Path

def check_project_structure():
    """Check if all required directories and files exist"""
    print("Checking project structure...")
    
    required_dirs = [
        "src/data_processing",
        "src/models", 
        "src/report_generation",
        "src/utils",
        "api",
        "configs",
        "scripts",
        "data/raw",
        "data/processed",
        "tests"
    ]
    
    required_files = [
        "requirements.txt",
        "README.md",
        "setup.py",
        "configs/model_config.yaml",
        "configs/api_config.yaml", 
        "configs/training_config.yaml",
        "src/models/image_classifier.py",
        "src/models/trainer.py",
        "src/data_processing/image_preprocessor.py",
        "src/data_processing/dataset_loader.py",
        "src/report_generation/langchain_pipeline.py",
        "api/main.py",
        "scripts/train_classifier.py",
        "scripts/evaluate_model.py"
    ]
    
    missing_dirs = []
    missing_files = []
    
    # Check directories
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    # Check files
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_dirs:
        print("Missing directories:")
        for d in missing_dirs:
            print(f"   - {d}")
    
    if missing_files:
        print("Missing files:")
        for f in missing_files:
            print(f"   - {f}")
    
    if not missing_dirs and not missing_files:
        print("All required files and directories present")
        return True
    
    return False

def check_configs():
    """Check configuration files"""
    print("\nChecking configuration files...")
    
    try:
        import yaml
        
        # Check model config
        with open("configs/model_config.yaml", 'r') as f:
            model_config = yaml.safe_load(f)
        
        pathologies = model_config.get('pathologies', [])
        print(f"Model config loaded - {len(pathologies)} pathologies defined")
        
        # Check training config
        with open("configs/training_config.yaml", 'r') as f:
            training_config = yaml.safe_load(f)
        
        batch_size = training_config.get('training', {}).get('batch_size', 0)
        print(f"Training config loaded - batch size: {batch_size}")
        
        # Check API config
        with open("configs/api_config.yaml", 'r') as f:
            api_config = yaml.safe_load(f)
        
        port = api_config.get('api', {}).get('port', 0)
        print(f"API config loaded - port: {port}")
        
        return True
        
    except ImportError:
        print("PyYAML not installed - install with: pip install pyyaml")
        return False
    except Exception as e:
        print(f"Config error: {e}")
        return False

def check_python_version():
    """Check Python version"""
    print("\nChecking Python version...")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"Python {version.major}.{version.minor}.{version.micro} (compatible)")
        return True
    else:
        print(f"Python {version.major}.{version.minor}.{version.micro} (requires Python 3.8+)")
        return False

def show_next_steps():
    """Show next steps for setup"""
    print("\nNext Steps:")
    print("=" * 40)
    
    print("1. Install dependencies:")
    print("   pip install -r requirements.txt")
    
    print("\n2. Set OpenAI API key:")
    print("   set OPENAI_API_KEY=your_key_here  # Windows")
    print("   export OPENAI_API_KEY=your_key_here  # Linux/Mac")
    
    print("\n3. Test with demo training:")
    print("   python train_demo.py")
    
    print("\n4. Download real datasets:")
    print("   python scripts/download_datasets.py --dataset chestx14")
    
    print("\n5. Train on real data:")
    print("   python scripts/train_classifier.py --dataset chestx14 --prepare_data")
    
    print("\n6. Start API server:")
    print("   python api/main.py")
    
    print("\nSee TRAINING_GUIDE.md for detailed instructions")

def main():
    print("AI Radiology System - Setup Verification")
    print("=" * 50)
    
    # Check project structure
    structure_ok = check_project_structure()
    
    # Check Python version
    python_ok = check_python_version()
    
    # Check configs
    config_ok = check_configs()
    
    # Summary
    print("\nSetup Summary:")
    print("-" * 30)
    print(f"Project Structure: {'OK' if structure_ok else 'Issues'}")
    print(f"Python Version: {'OK' if python_ok else 'Issues'}")
    print(f"Configuration: {'OK' if config_ok else 'Issues'}")
    
    if structure_ok and python_ok:
        print("\nProject setup verified successfully!")
        show_next_steps()
        return 0
    else:
        print("\nPlease fix the issues above before proceeding")
        return 1

if __name__ == "__main__":
    exit(main())