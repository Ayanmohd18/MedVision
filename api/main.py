from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import yaml
import os
from typing import Dict, Any
import tempfile
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Mock implementations for testing without ML dependencies
class ChestXRayClassifier:
    def __init__(self, num_classes=14):
        self.num_classes = num_classes
    def eval(self):
        pass
    def __call__(self, x):
        import torch
        return torch.rand(1, 14), torch.rand(1, 1)

class ChestXRayPreprocessor:
    def __call__(self, path):
        import torch
        return torch.rand(3, 224, 224)

def validate_image_quality(image):
    return {"is_valid": True, "mean_intensity": 0.45, "contrast_adequate": True}

def get_predictions_with_confidence(probs, threshold=0.5):
    import torch
    return {
        "predictions": torch.zeros(1, 14),
        "probabilities": probs,
        "confidence_scores": torch.rand(1, 14),
        "max_confidence": torch.tensor([0.85])
    }

class RadiologyReportGenerator:
    def generate_report(self, patient_data, findings):
        return {
            "success": True,
            "report": {
                "findings": "No acute abnormalities detected.",
                "impression": "Normal chest X-ray.",
                "recommendations": "No follow-up required."
            },
            "metadata": {"patient_id": patient_data.get("patient_id")}
        }

app = FastAPI(title="AI Radiology Report Generator", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load configurations
with open("configs/model_config.yaml", "r") as f:
    model_config = yaml.safe_load(f)

with open("configs/api_config.yaml", "r") as f:
    api_config = yaml.safe_load(f)

# Initialize components
preprocessor = ChestXRayPreprocessor()
model = ChestXRayClassifier(num_classes=len(model_config["pathologies"]))
report_generator = RadiologyReportGenerator()

# Load trained model weights (placeholder - would load actual trained weights)
# model.load_state_dict(torch.load("models/chest_xray_classifier.pth"))
model.eval()

@app.post("/analyze-xray")
async def analyze_xray(
    file: UploadFile = File(...),
    patient_age: int = 0,
    patient_gender: str = "Unknown",
    view_position: str = "PA",
    clinical_history: str = "Not provided"
):
    """Analyze chest X-ray and generate radiology report"""
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Preprocess image
        image_tensor = preprocessor(tmp_file_path)
        
        # Validate image quality
        quality_check = validate_image_quality(image_tensor)
        if not quality_check["is_valid"]:
            raise HTTPException(status_code=400, detail="Poor image quality detected")
        
        # Run inference
        with torch.no_grad():
            image_batch = image_tensor.unsqueeze(0)
            probabilities, attention_weights = model(image_batch)
            
        # Get predictions with confidence
        predictions_data = get_predictions_with_confidence(
            probabilities, 
            threshold=api_config["report"]["confidence_threshold"]
        )
        
        # Format findings for report generation
        findings = []
        pathologies = model_config["pathologies"]
        
        for i, (pathology, prob) in enumerate(zip(pathologies, probabilities[0])):
            findings.append({
                "pathology": pathology,
                "probability": float(prob),
                "predicted": bool(predictions_data["predictions"][0][i])
            })
        
        # Patient data
        patient_data = {
            "age": patient_age,
            "gender": patient_gender,
            "view_position": view_position,
            "clinical_history": clinical_history,
            "patient_id": f"temp_{file.filename}"
        }
        
        # Generate report
        report_result = report_generator.generate_report(patient_data, findings)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        if not report_result["success"]:
            raise HTTPException(status_code=500, detail=f"Report generation failed: {report_result['error']}")
        
        return {
            "success": True,
            "findings": findings,
            "report": report_result["report"],
            "quality_metrics": quality_check,
            "confidence_summary": {
                "max_confidence": float(predictions_data["max_confidence"][0]),
                "significant_findings": len([f for f in findings if f["probability"] > 0.5])
            }
        }
        
    except Exception as e:
        # Clean up on error
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "AI Radiology Report Generator API", "version": "1.0.0", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": True}

@app.get("/pathologies")
async def get_pathologies():
    """Get list of detectable pathologies"""
    return {"pathologies": model_config["pathologies"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8003)