import pytest
import torch
import numpy as np
from PIL import Image
import tempfile
import os

# Add src to path
import sys
sys.path.append('../src')

from src.data_processing.image_preprocessor import ChestXRayPreprocessor, validate_image_quality
from src.models.image_classifier import ChestXRayClassifier, get_predictions_with_confidence
from src.report_generation.prompt_templates import create_findings_text

class TestSystemIntegration:
    
    def setup_method(self):
        """Setup test fixtures"""
        self.preprocessor = ChestXRayPreprocessor()
        self.model = ChestXRayClassifier(num_classes=14)
        self.model.eval()
        
        # Create dummy image
        self.test_image = Image.new('RGB', (512, 512), color='gray')
        
    def test_image_preprocessing(self):
        """Test image preprocessing pipeline"""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            self.test_image.save(tmp_file.name)
            
            # Test preprocessing
            processed = self.preprocessor(tmp_file.name)
            
            assert isinstance(processed, torch.Tensor)
            assert processed.shape == (3, 224, 224)  # RGB, 224x224
            
            # Test quality validation
            quality = validate_image_quality(processed)
            assert 'is_valid' in quality
            assert 'mean_intensity' in quality
            
            os.unlink(tmp_file.name)
    
    def test_model_inference(self):
        """Test model inference"""
        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            probabilities, attention = self.model(dummy_input)
        
        assert probabilities.shape == (1, 14)  # Batch size 1, 14 classes
        assert torch.all(probabilities >= 0) and torch.all(probabilities <= 1)  # Valid probabilities
        
        # Test prediction conversion
        predictions = get_predictions_with_confidence(probabilities)
        assert 'predictions' in predictions
        assert 'confidence_scores' in predictions
    
    def test_findings_formatting(self):
        """Test findings text generation"""
        pathologies = ["Pneumonia", "Cardiomegaly", "Atelectasis"]
        probabilities = [0.8, 0.6, 0.3]
        
        findings_text = create_findings_text(pathologies, probabilities, confidence_threshold=0.5)
        
        assert "Pneumonia" in findings_text
        assert "Cardiomegaly" in findings_text
        assert "Atelectasis" not in findings_text  # Below threshold
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline without OpenAI API"""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            self.test_image.save(tmp_file.name)
            
            # Preprocess
            processed_image = self.preprocessor(tmp_file.name)
            
            # Model inference
            with torch.no_grad():
                probabilities, _ = self.model(processed_image.unsqueeze(0))
            
            # Format results
            pathologies = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
                          "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
                          "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
            
            findings = []
            for i, (pathology, prob) in enumerate(zip(pathologies, probabilities[0])):
                findings.append({
                    "pathology": pathology,
                    "probability": float(prob),
                    "predicted": bool(prob > 0.5)
                })
            
            assert len(findings) == 14
            assert all(0 <= f["probability"] <= 1 for f in findings)
            
            os.unlink(tmp_file.name)

if __name__ == "__main__":
    pytest.main([__file__])