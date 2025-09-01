#!/usr/bin/env python3
"""
Demo script for AI Radiology Report Generation System
"""

import torch
import numpy as np
from PIL import Image
import tempfile
import os
import sys

# Add src to path
sys.path.append('src')

from src.data_processing.image_preprocessor import ChestXRayPreprocessor, validate_image_quality
from src.models.image_classifier import ChestXRayClassifier, get_predictions_with_confidence
from src.report_generation.prompt_templates import create_findings_text
from src.utils.config import config

def create_demo_image():
    """Create a demo chest X-ray-like image"""
    # Create a grayscale image that resembles a chest X-ray
    img_array = np.random.normal(0.3, 0.1, (512, 512))
    img_array = np.clip(img_array, 0, 1)
    
    # Add some structure to make it more X-ray-like
    # Simulate ribcage
    for i in range(50, 450, 40):
        img_array[i:i+3, 100:400] *= 0.7
    
    # Simulate heart shadow
    center_x, center_y = 256, 200
    for i in range(512):
        for j in range(512):
            dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            if dist < 80:
                img_array[i, j] *= 0.8
    
    # Convert to PIL Image
    img_array = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img_array, mode='L').convert('RGB')

def run_demo():
    """Run complete demo pipeline"""
    print("🏥 AI Radiology Report Generation System Demo")
    print("=" * 50)
    
    # Initialize components
    print("📋 Initializing components...")
    preprocessor = ChestXRayPreprocessor()
    model = ChestXRayClassifier(num_classes=14)
    model.eval()
    
    pathologies = config.model_config['pathologies']
    print(f"✅ Model loaded with {len(pathologies)} pathology classes")
    
    # Create demo image
    print("\n🖼️  Creating demo chest X-ray image...")
    demo_image = create_demo_image()
    
    # Save temporarily
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
        demo_image.save(tmp_file.name)
        tmp_path = tmp_file.name
    
    try:
        # Preprocess image
        print("🔄 Preprocessing image...")
        processed_image = preprocessor(tmp_path)
        
        # Validate quality
        quality_check = validate_image_quality(processed_image)
        print(f"📊 Image quality: {'✅ Valid' if quality_check['is_valid'] else '❌ Invalid'}")
        print(f"   Mean intensity: {quality_check['mean_intensity']:.3f}")
        print(f"   Contrast adequate: {'✅ Yes' if quality_check['contrast_adequate'] else '❌ No'}")
        
        # Run model inference
        print("\n🧠 Running AI analysis...")
        with torch.no_grad():
            probabilities, attention_weights = model(processed_image.unsqueeze(0))
        
        # Get predictions
        predictions_data = get_predictions_with_confidence(probabilities, threshold=0.5)
        
        # Format findings
        findings = []
        significant_findings = []
        
        for i, (pathology, prob) in enumerate(zip(pathologies, probabilities[0])):
            findings.append({
                "pathology": pathology,
                "probability": float(prob),
                "predicted": bool(predictions_data["predictions"][0][i])
            })
            
            if prob > 0.3:  # Show findings above 30% confidence
                significant_findings.append((pathology, float(prob)))
        
        # Display results
        print("\n📋 Analysis Results:")
        print("-" * 30)
        
        if significant_findings:
            print("🔍 Detected findings:")
            for pathology, prob in sorted(significant_findings, key=lambda x: x[1], reverse=True):
                confidence_level = "🔴 High" if prob > 0.7 else "🟡 Moderate" if prob > 0.5 else "🟢 Low"
                print(f"   • {pathology}: {prob:.1%} ({confidence_level})")
        else:
            print("✅ No significant abnormalities detected")
        
        print(f"\n📈 Overall confidence: {float(predictions_data['max_confidence'][0]):.1%}")
        
        # Generate findings text
        findings_text = create_findings_text(
            [f["pathology"] for f in findings],
            [f["probability"] for f in findings],
            confidence_threshold=0.5
        )
        
        print("\n📝 Generated Findings Summary:")
        print("-" * 40)
        print(findings_text)
        
        print("\n💡 Note: This is a demo with a synthetic image and untrained model.")
        print("   In production, use real chest X-rays and trained models.")
        
    finally:
        # Clean up
        os.unlink(tmp_path)
    
    print("\n✅ Demo completed successfully!")

if __name__ == "__main__":
    try:
        run_demo()
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        sys.exit(1)