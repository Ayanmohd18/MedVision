#!/usr/bin/env python3
"""
Dataset download and setup script
"""

import os
import requests
import zipfile
from pathlib import Path
import argparse

def download_chestx_ray14():
    """Download ChestX-ray14 dataset"""
    print("📥 Downloading ChestX-ray14 dataset...")
    
    # Create directory
    data_dir = Path("data/raw/chestx_ray14")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset URLs (these are example URLs - replace with actual NIH URLs)
    urls = {
        "Data_Entry_2017.csv": "https://nihcc.app.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.csv",
        # Add image download URLs here
    }
    
    for filename, url in urls.items():
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"  Downloading {filename}...")
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        
                print(f"  ✅ {filename} downloaded")
            except Exception as e:
                print(f"  ❌ Failed to download {filename}: {e}")
        else:
            print(f"  ✅ {filename} already exists")
    
    print("📋 ChestX-ray14 metadata downloaded. Please download image files manually from:")
    print("   https://nihcc.app.box.com/v/ChestXray-NIHCC")

def download_mimic_cxr():
    """Setup MIMIC-CXR dataset (requires PhysioNet access)"""
    print("📥 Setting up MIMIC-CXR dataset...")
    
    data_dir = Path("data/raw/mimic_cxr")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("📋 MIMIC-CXR requires PhysioNet credentialed access.")
    print("   1. Complete CITI training: https://physionet.org/about/citi-course/")
    print("   2. Request access: https://physionet.org/content/mimic-cxr/2.0.0/")
    print("   3. Download using wget with your credentials:")
    print("      wget -r -N -c -np --user YOUR_USERNAME --ask-password https://physionet.org/files/mimic-cxr/2.0.0/")

def setup_sample_data():
    """Create sample data for testing"""
    print("📝 Creating sample data for testing...")
    
    import pandas as pd
    import numpy as np
    from PIL import Image
    
    # Create sample ChestX-ray14 CSV
    sample_data = {
        'Image Index': [f'sample_{i:04d}.png' for i in range(100)],
        'Finding Labels': np.random.choice([
            'No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion',
            'Infiltration', 'Mass', 'Nodule', 'Pneumonia'
        ], 100),
        'Patient ID': [f'P{i//4:05d}' for i in range(100)],
        'Patient Age': np.random.randint(20, 80, 100),
        'Patient Gender': np.random.choice(['M', 'F'], 100),
        'View Position': np.random.choice(['PA', 'AP'], 100)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Save sample CSV
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/sample_data.csv", index=False)
    
    # Create sample images
    sample_img_dir = Path("data/raw/sample_images")
    sample_img_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(10):  # Create 10 sample images
        # Create synthetic chest X-ray-like image
        img_array = np.random.normal(0.3, 0.1, (512, 512))
        img_array = np.clip(img_array, 0, 1)
        
        # Add some structure
        for j in range(50, 450, 40):
            img_array[j:j+3, 100:400] *= 0.7
        
        img_array = (img_array * 255).astype(np.uint8)
        img = Image.fromarray(img_array, mode='L')
        img.save(sample_img_dir / f'sample_{i:04d}.png')
    
    print(f"✅ Created {len(df)} sample records and 10 sample images")
    print("   Use this for testing the training pipeline")

def main():
    parser = argparse.ArgumentParser(description='Download and setup datasets')
    parser.add_argument('--dataset', choices=['chestx14', 'mimic', 'sample', 'all'],
                       default='sample', help='Dataset to download/setup')
    
    args = parser.parse_args()
    
    print("📦 Dataset Download and Setup")
    print("=" * 40)
    
    if args.dataset in ['chestx14', 'all']:
        download_chestx_ray14()
    
    if args.dataset in ['mimic', 'all']:
        download_mimic_cxr()
    
    if args.dataset in ['sample', 'all']:
        setup_sample_data()
    
    print("\n✅ Dataset setup completed!")

if __name__ == "__main__":
    main()