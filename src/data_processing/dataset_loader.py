import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from typing import Dict, Tuple, Optional
from .image_preprocessor import ChestXRayPreprocessor

class ChestXRayDataset(Dataset):
    def __init__(self, csv_file: str, image_dir: str, transform=None, pathologies=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform or ChestXRayPreprocessor()
        self.pathologies = pathologies or [
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
            "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
            "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
        ]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        row = self.data.iloc[idx]
        
        # Load image
        image_path = os.path.join(self.image_dir, row['Image Index'])
        image = self.transform(image_path)
        
        # Create multi-label target
        labels = torch.zeros(len(self.pathologies))
        for i, pathology in enumerate(self.pathologies):
            if pathology in row['Finding Labels']:
                labels[i] = 1.0
        
        # Metadata
        metadata = {
            'patient_id': row.get('Patient ID', ''),
            'age': row.get('Patient Age', 0),
            'gender': row.get('Patient Gender', ''),
            'view_position': row.get('View Position', ''),
            'image_index': row['Image Index']
        }
        
        return image, labels, metadata

def create_data_loaders(train_csv: str, val_csv: str, image_dir: str, 
                       batch_size: int = 32, num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders"""
    
    train_dataset = ChestXRayDataset(train_csv, image_dir)
    val_dataset = ChestXRayDataset(val_csv, image_dir)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader