#!/usr/bin/env python3
"""
Demo training script with sample data
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from PIL import Image

# Add src to path
sys.path.append('src')

from src.models.trainer import ModelTrainer
from src.data_processing.data_splitter import prepare_chestx_ray14

def create_sample_dataset():
    """Create sample dataset for demo training"""
    print("📝 Creating sample dataset...")
    
    # Create directories
    os.makedirs("data/raw/sample_images", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # Generate sample data
    pathologies = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
        "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
        "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
    ]
    
    # Create sample CSV data
    n_samples = 200
    sample_data = []
    
    for i in range(n_samples):
        # Random pathology assignment
        finding_labels = []
        for pathology in pathologies:
            if np.random.random() < 0.1:  # 10% chance for each pathology
                finding_labels.append(pathology)
        
        if not finding_labels:
            finding_labels = ["No Finding"]
        
        sample_data.append({
            'Image Index': f'sample_{i:04d}.png',
            'Finding Labels': '|'.join(finding_labels),
            'Patient ID': f'P{i//4:05d}',
            'Patient Age': np.random.randint(20, 80),
            'Patient Gender': np.random.choice(['M', 'F']),
            'View Position': np.random.choice(['PA', 'AP'])
        })
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Add binary labels for each pathology
    for pathology in pathologies:
        df[pathology] = df['Finding Labels'].str.contains(pathology).astype(int)
    
    # Split data
    train_size = int(0.7 * len(df))
    val_size = int(0.15 * len(df))
    
    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]
    
    # Save splits
    train_df.to_csv("data/processed/train_sample.csv", index=False)
    val_df.to_csv("data/processed/val_sample.csv", index=False)
    test_df.to_csv("data/processed/test_sample.csv", index=False)
    
    # Create sample images
    print("🖼️  Generating sample chest X-ray images...")
    for i in range(min(50, n_samples)):  # Create 50 sample images
        # Create synthetic chest X-ray-like image
        img_array = np.random.normal(0.3, 0.1, (512, 512))
        img_array = np.clip(img_array, 0, 1)
        
        # Add some chest X-ray-like structure
        # Simulate ribcage
        for j in range(50, 450, 40):
            img_array[j:j+3, 100:400] *= 0.7
        
        # Simulate heart shadow
        center_x, center_y = 256, 200
        for x in range(512):
            for y in range(512):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist < 80:
                    img_array[x, y] *= 0.8
        
        # Add some random pathology-like features
        if np.random.random() < 0.3:  # 30% chance of adding "abnormality"
            # Add bright spot (could represent mass/nodule)
            spot_x, spot_y = np.random.randint(100, 400, 2)
            for x in range(max(0, spot_x-10), min(512, spot_x+10)):
                for y in range(max(0, spot_y-10), min(512, spot_y+10)):
                    img_array[x, y] = min(1.0, img_array[x, y] * 1.3)
        
        # Convert to PIL Image and save
        img_array = (img_array * 255).astype(np.uint8)
        img = Image.fromarray(img_array, mode='L').convert('RGB')
        img.save(f"data/raw/sample_images/sample_{i:04d}.png")
    
    print(f"✅ Created {len(df)} sample records and 50 sample images")
    return train_df, val_df, test_df

def run_demo_training():
    """Run demo training with sample data"""
    print("🏥 Demo Training - AI Radiology System")
    print("=" * 50)
    
    # Create sample dataset
    train_df, val_df, test_df = create_sample_dataset()
    
    # Initialize trainer with reduced epochs for demo
    trainer = ModelTrainer(config_path="configs/training_config.yaml")
    
    # Override config for demo (shorter training)
    trainer.config['training']['num_epochs'] = 5
    trainer.config['training']['batch_size'] = 8
    trainer.config['training']['patience'] = 3
    
    print(f"📊 Dataset sizes:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    
    # Training paths
    train_csv = "data/processed/train_sample.csv"
    val_csv = "data/processed/val_sample.csv"
    image_dir = "data/raw/sample_images"
    
    print(f"\n🚀 Starting demo training (5 epochs)...")
    print("   Note: This is a demo with synthetic data")
    
    try:
        # Train model
        best_auc = trainer.train(train_csv, val_csv, image_dir)
        
        print(f"\n✅ Demo training completed!")
        print(f"🏆 Best validation AUC: {best_auc:.4f}")
        
        # Save model
        os.makedirs("models", exist_ok=True)
        if os.path.exists("models/checkpoints/best_model.pth"):
            import shutil
            shutil.copy(
                "models/checkpoints/best_model.pth",
                "models/chest_xray_classifier_demo.pth"
            )
            print("💾 Demo model saved to models/chest_xray_classifier_demo.pth")
        
        print("\n📋 Next steps:")
        print("  1. Download real ChestX-ray14 dataset")
        print("  2. Run: python scripts/download_datasets.py --dataset chestx14")
        print("  3. Run: python scripts/train_classifier.py --prepare_data --dataset chestx14")
        
    except Exception as e:
        print(f"❌ Demo training failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(run_demo_training())