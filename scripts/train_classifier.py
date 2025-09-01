#!/usr/bin/env python3
"""
Training script for chest X-ray classifier
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.models.trainer import ModelTrainer
from src.data_processing.data_splitter import prepare_chestx_ray14, prepare_mimic_cxr

def main():
    parser = argparse.ArgumentParser(description='Train chest X-ray classifier')
    parser.add_argument('--dataset', choices=['chestx14', 'mimic', 'both'], 
                       default='chestx14', help='Dataset to use for training')
    parser.add_argument('--data_dir', type=str, default='data/raw',
                       help='Root directory containing datasets')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                       help='Training configuration file')
    parser.add_argument('--prepare_data', action='store_true',
                       help='Prepare dataset splits before training')
    
    args = parser.parse_args()
    
    print("🏥 Starting Chest X-Ray Classifier Training")
    print("=" * 50)
    
    # Prepare data if requested
    if args.prepare_data:
        print("📊 Preparing dataset splits...")
        
        if args.dataset in ['chestx14', 'both']:
            chestx14_dir = os.path.join(args.data_dir, 'chestx_ray14')
            if os.path.exists(chestx14_dir):
                prepare_chestx_ray14(chestx14_dir)
            else:
                print(f"⚠️  ChestX-ray14 directory not found: {chestx14_dir}")
        
        if args.dataset in ['mimic', 'both']:
            mimic_dir = os.path.join(args.data_dir, 'mimic_cxr')
            if os.path.exists(mimic_dir):
                prepare_mimic_cxr(mimic_dir)
            else:
                print(f"⚠️  MIMIC-CXR directory not found: {mimic_dir}")
    
    # Initialize trainer
    trainer = ModelTrainer(config_path=args.config)
    
    # Training paths
    if args.dataset == 'chestx14':
        train_csv = "data/processed/train_chestx14.csv"
        val_csv = "data/processed/val_chestx14.csv"
        image_dir = os.path.join(args.data_dir, "chestx_ray14/images")
        
    elif args.dataset == 'mimic':
        train_csv = "data/processed/train_mimic.csv"
        val_csv = "data/processed/val_mimic.csv"
        image_dir = os.path.join(args.data_dir, "mimic_cxr/files")
        
    else:  # both datasets
        print("⚠️  Combined training not implemented yet. Using ChestX-ray14.")
        train_csv = "data/processed/train_chestx14.csv"
        val_csv = "data/processed/val_chestx14.csv"
        image_dir = os.path.join(args.data_dir, "chestx_ray14/images")
    
    # Check if files exist
    if not os.path.exists(train_csv):
        print(f"❌ Training CSV not found: {train_csv}")
        print("   Run with --prepare_data flag to create dataset splits")
        return
    
    if not os.path.exists(image_dir):
        print(f"❌ Image directory not found: {image_dir}")
        print("   Please download and extract the dataset")
        return
    
    print(f"📁 Using dataset: {args.dataset}")
    print(f"📁 Training CSV: {train_csv}")
    print(f"📁 Validation CSV: {val_csv}")
    print(f"📁 Image directory: {image_dir}")
    
    # Start training
    print("\n🚀 Starting training...")
    try:
        best_auc = trainer.train(train_csv, val_csv, image_dir)
        print(f"\n✅ Training completed successfully!")
        print(f"🏆 Best validation AUC: {best_auc:.4f}")
        
        # Copy best model to main models directory
        os.makedirs("models", exist_ok=True)
        import shutil
        shutil.copy(
            "models/checkpoints/best_model.pth",
            "models/chest_xray_classifier.pth"
        )
        print("💾 Best model saved to models/chest_xray_classifier.pth")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())