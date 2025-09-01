import torch
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, ScaleIntensity,
    Resize, NormalizeIntensity, ToTensor
)
import numpy as np
from typing import Dict, Any

class ChestXRayPreprocessor:
    def __init__(self, image_size=(224, 224), normalize_mean=None, normalize_std=None):
        self.image_size = image_size
        self.normalize_mean = normalize_mean or [0.485, 0.456, 0.406]
        self.normalize_std = normalize_std or [0.229, 0.224, 0.225]
        
        self.transforms = Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            ScaleIntensity(minv=0.0, maxv=1.0),
            Resize(spatial_size=self.image_size),
            NormalizeIntensity(subtrahend=self.normalize_mean, divisor=self.normalize_std),
            ToTensor()
        ])
    
    def __call__(self, image_path: str) -> torch.Tensor:
        return self.transforms(image_path)
    
    def preprocess_batch(self, image_paths: list) -> torch.Tensor:
        processed_images = []
        for path in image_paths:
            img = self(path)
            processed_images.append(img)
        return torch.stack(processed_images)

def validate_image_quality(image: torch.Tensor, min_intensity=0.01, max_intensity=0.99) -> Dict[str, Any]:
    """Validate chest X-ray image quality"""
    mean_intensity = torch.mean(image).item()
    std_intensity = torch.std(image).item()
    
    quality_checks = {
        'mean_intensity': mean_intensity,
        'std_intensity': std_intensity,
        'is_valid': min_intensity < mean_intensity < max_intensity and std_intensity > 0.01,
        'contrast_adequate': std_intensity > 0.05
    }
    
    return quality_checks