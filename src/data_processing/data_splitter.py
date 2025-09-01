import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import os

def prepare_chestx_ray14(data_dir: str, output_dir: str = "data/processed"):
    """Prepare ChestX-ray14 dataset with train/val/test splits"""
    
    # Load metadata
    csv_path = os.path.join(data_dir, "Data_Entry_2017.csv")
    df = pd.read_csv(csv_path)
    
    # Clean data
    df = df.dropna(subset=['Image Index', 'Finding Labels'])
    
    # Create binary labels for each pathology
    pathologies = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
        "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
        "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
    ]
    
    for pathology in pathologies:
        df[pathology] = df['Finding Labels'].str.contains(pathology).astype(int)
    
    # Split by patient to avoid data leakage
    unique_patients = df['Patient ID'].unique()
    
    train_patients, temp_patients = train_test_split(
        unique_patients, test_size=0.3, random_state=42
    )
    val_patients, test_patients = train_test_split(
        temp_patients, test_size=0.5, random_state=42
    )
    
    # Create splits
    train_df = df[df['Patient ID'].isin(train_patients)]
    val_df = df[df['Patient ID'].isin(val_patients)]
    test_df = df[df['Patient ID'].isin(test_patients)]
    
    # Save splits
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train_chestx14.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val_chestx14.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_chestx14.csv"), index=False)
    
    print(f"ChestX-ray14 splits created:")
    print(f"  Train: {len(train_df)} images ({len(train_patients)} patients)")
    print(f"  Val: {len(val_df)} images ({len(val_patients)} patients)")
    print(f"  Test: {len(test_df)} images ({len(test_patients)} patients)")
    
    return train_df, val_df, test_df

def prepare_mimic_cxr(data_dir: str, output_dir: str = "data/processed"):
    """Prepare MIMIC-CXR dataset with train/val/test splits"""
    
    # Load metadata
    csv_path = os.path.join(data_dir, "mimic-cxr-2.0.0-metadata.csv")
    if not os.path.exists(csv_path):
        print(f"MIMIC-CXR metadata not found at {csv_path}")
        return None, None, None
    
    df = pd.read_csv(csv_path)
    
    # Filter for PA/AP views only
    df = df[df['ViewPosition'].isin(['PA', 'AP'])]
    
    # Use existing train/val/test splits if available
    if 'split' in df.columns:
        train_df = df[df['split'] == 'train']
        val_df = df[df['split'] == 'validate']
        test_df = df[df['split'] == 'test']
    else:
        # Create splits by subject
        unique_subjects = df['subject_id'].unique()
        train_subjects, temp_subjects = train_test_split(
            unique_subjects, test_size=0.3, random_state=42
        )
        val_subjects, test_subjects = train_test_split(
            temp_subjects, test_size=0.5, random_state=42
        )
        
        train_df = df[df['subject_id'].isin(train_subjects)]
        val_df = df[df['subject_id'].isin(val_subjects)]
        test_df = df[df['subject_id'].isin(test_subjects)]
    
    # Save splits
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train_mimic.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val_mimic.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_mimic.csv"), index=False)
    
    print(f"MIMIC-CXR splits created:")
    print(f"  Train: {len(train_df)} images")
    print(f"  Val: {len(val_df)} images")
    print(f"  Test: {len(test_df)} images")
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    # Prepare datasets
    prepare_chestx_ray14("data/raw/chestx_ray14")
    prepare_mimic_cxr("data/raw/mimic_cxr")