

# 🩺 MedVision – AI-Assisted Radiology Report Generation

MedVision is an end-to-end AI system that transforms chest X-ray images into **structured radiology reports**. It combines **medical imaging, deep learning, and natural language generation** to support radiologists in faster and more consistent diagnoses.

---

## ✨ Key Features

- 🔬 **Medical Image Processing** – MONAI-powered preprocessing tailored for chest X-rays  
- 🧠 **Pathology Detection** – Multi-label classification of **14 thoracic pathologies** using PyTorch  
- 📑 **AI Report Generation** – GPT-4 + LangChain for structured and context-aware reporting  
- ⚡ **FastAPI REST Service** – Easy-to-integrate endpoints for deployment  
- ✅ **Quality Validation** – Automated checks for image integrity and report accuracy  

---

## 🚀 Quick Start

### 1️⃣ Installation

```bash
# Clone repository
git clone https://github.com/Ayanmohd18/MedVision.git
cd MedVision

# Create virtual environment
python -m venv medvision_env
medvision_env\Scripts\activate   # Windows
# source medvision_env/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Configuration

1. Set your **OpenAI API Key**  
   ```bash
   set OPENAI_API_KEY=your_api_key_here
   ```
2. Adjust model configs → `configs/model_config.yaml`  
3. Adjust API configs → `configs/api_config.yaml`  

### 3️⃣ Running the API

```bash
python api/main.py
```

API will be live at: **http://localhost:8000**

---

## 🔗 API Endpoints

| Method | Endpoint            | Description                          |
|--------|---------------------|--------------------------------------|
| POST   | `/analyze-xray`     | Upload and analyze chest X-ray       |
| GET    | `/health`           | API health check                     |
| GET    | `/pathologies`      | List supported pathologies           |

---

## 📂 Project Structure

```
MedVision/
├── data/                     # Datasets (ChestX-ray14, MIMIC-CXR, etc.)
├── src/
│   ├── data_processing/      # Preprocessing & loaders
│   ├── models/              # PyTorch classifiers
│   ├── report_generation/   # LangChain + GPT-4 reporting
│   └── utils/               # Helper functions
├── api/                     # FastAPI app
├── configs/                 # Config files
├── tests/                   # Test suite
└── scripts/                 # Training / utility scripts
```

---

## 🩻 Supported Pathologies

- Atelectasis  
- Cardiomegaly  
- Effusion  
- Infiltration  
- Mass  
- Nodule  
- Pneumonia  
- Pneumothorax  
- Consolidation  
- Edema  
- Emphysema  
- Fibrosis  
- Pleural Thickening  
- Hernia  

---

## 🛠 Development

### Train Models
```bash
# Prepare dataset
python src/data_processing/dataset_loader.py

# Train multi-label classifier
python scripts/train_classifier.py
```

### Run Tests
```bash
pytest tests/
```

---

## ⚠️ Disclaimer
MedVision is intended for **research and educational purposes only**.  
It is **not a certified medical device** and should not be used for clinical decision-making without regulatory approval.

---

## 📜 License
MIT License – see [LICENSE](LICENSE) for details.
