# 🏥 Training Guide - AI Radiology System

This guide walks you through training the chest X-ray classifier on real datasets.

## 📋 Prerequisites

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Set Environment Variables**
```bash
# Windows
set OPENAI_API_KEY=your_openai_api_key_here

# Linux/Mac
export OPENAI_API_KEY=your_openai_api_key_here
```

## 🚀 Quick Start - Demo Training

Test the training pipeline with synthetic data:

```bash
python train_demo.py
```

This creates sample data and trains for 5 epochs to verify everything works.

## 📊 Real Dataset Training

### Step 1: Download Datasets

#### ChestX-ray14 Dataset
```bash
# Download metadata and setup
python scripts/download_datasets.py --dataset chestx14

# Manual download required for images:
# Visit: https://nihcc.app.box.com/v/ChestXray-NIHCC
# Download all image files to: data/raw/chestx_ray14/images/
```

#### MIMIC-CXR Dataset (Optional)
```bash
# Setup MIMIC-CXR (requires PhysioNet access)
python scripts/download_datasets.py --dataset mimic

# Follow instructions to get PhysioNet access
# Download to: data/raw/mimic_cxr/
```

### Step 2: Prepare Data Splits

```bash
# Prepare ChestX-ray14 splits
python scripts/train_classifier.py --dataset chestx14 --prepare_data

# Or prepare MIMIC-CXR splits
python scripts/train_classifier.py --dataset mimic --prepare_data
```

### Step 3: Train Model

```bash
# Train on ChestX-ray14
python scripts/train_classifier.py --dataset chestx14

# Train on MIMIC-CXR
python scripts/train_classifier.py --dataset mimic

# Custom configuration
python scripts/train_classifier.py --dataset chestx14 --config configs/training_config.yaml
```

### Step 4: Evaluate Model

```bash
# Evaluate trained model
python scripts/evaluate_model.py --model models/chest_xray_classifier.pth

# Custom evaluation
python scripts/evaluate_model.py \
    --model models/chest_xray_classifier.pth \
    --test_csv data/processed/test_chestx14.csv \
    --image_dir data/raw/chestx_ray14/images \
    --output_dir results
```

## ⚙️ Configuration

### Training Configuration (`configs/training_config.yaml`)

```yaml
training:
  batch_size: 32          # Adjust based on GPU memory
  learning_rate: 0.001    # Learning rate
  num_epochs: 50          # Maximum epochs
  patience: 10            # Early stopping patience
  
datasets:
  chestx_ray14:
    data_dir: "data/raw/chestx_ray14"
    csv_file: "Data_Entry_2017.csv"
    image_dir: "images"
```

### Model Configuration (`configs/model_config.yaml`)

```yaml
model:
  image_classifier:
    backbone: "resnet50"    # Model backbone
    num_classes: 14         # Number of pathologies
    dropout_rate: 0.3       # Dropout rate
```

## 📈 Expected Results

### ChestX-ray14 Benchmarks
- **Mean AUC**: ~0.75-0.85 (varies by pathology)
- **Training Time**: 4-8 hours on GPU
- **Best Pathologies**: Cardiomegaly, Effusion
- **Challenging**: Hernia, Fibrosis

### Performance by Pathology
| Pathology | Expected AUC | Notes |
|-----------|--------------|-------|
| Cardiomegaly | 0.85-0.90 | Usually highest performance |
| Effusion | 0.80-0.85 | Good performance |
| Pneumonia | 0.70-0.80 | Moderate performance |
| Atelectasis | 0.70-0.75 | Challenging |
| Hernia | 0.60-0.70 | Most challenging |

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size` in training config
   - Use smaller image size

2. **Dataset Not Found**
   - Verify dataset paths in config
   - Run data preparation scripts

3. **Low Performance**
   - Increase training epochs
   - Adjust learning rate
   - Check data quality

### GPU Requirements

- **Minimum**: 8GB VRAM (batch_size=16)
- **Recommended**: 16GB+ VRAM (batch_size=32+)
- **CPU Training**: Possible but very slow

## 📊 Monitoring Training

### Using TensorBoard (Optional)
```bash
# Install tensorboard
pip install tensorboard

# Monitor training (if implemented)
tensorboard --logdir models/logs
```

### Training Logs
- Checkpoints saved to: `models/checkpoints/`
- Best model: `models/chest_xray_classifier.pth`
- Training logs: Console output

## 🎯 Production Deployment

After training:

1. **Update API Model Path**
```python
# In api/main.py
model.load_state_dict(torch.load("models/chest_xray_classifier.pth"))
```

2. **Test API with Trained Model**
```bash
python api/main.py
```

3. **Validate Performance**
```bash
python scripts/evaluate_model.py
```

## 📚 Dataset Information

### ChestX-ray14
- **Size**: 112,120 images from 30,805 patients
- **Classes**: 14 pathologies + "No Finding"
- **Format**: PNG images, CSV metadata
- **License**: NIH Clinical Center

### MIMIC-CXR
- **Size**: 377,110 images from 65,079 patients
- **Classes**: Multiple pathologies with reports
- **Format**: DICOM/JPG images, CSV metadata
- **License**: PhysioNet Credentialed Health Data

## 🔬 Advanced Training

### Multi-GPU Training
```bash
# Use DataParallel (modify trainer.py)
model = nn.DataParallel(model)
```

### Transfer Learning
```bash
# Start from pre-trained weights
python scripts/train_classifier.py --pretrained path/to/weights.pth
```

### Hyperparameter Tuning
- Experiment with learning rates: [0.0001, 0.001, 0.01]
- Try different backbones: ResNet50, DenseNet121, EfficientNet
- Adjust dropout rates: [0.2, 0.3, 0.5]

## 📞 Support

For issues:
1. Check this guide first
2. Verify dataset setup
3. Check GPU/memory requirements
4. Review error logs

Training typically takes 4-8 hours on modern GPUs for full ChestX-ray14 dataset.