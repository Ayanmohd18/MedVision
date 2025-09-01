#!/usr/bin/env python3
"""
Model evaluation script
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.models.trainer import load_trained_model
from src.data_processing.dataset_loader import ChestXRayDataset
from torch.utils.data import DataLoader

def evaluate_model(model_path, test_csv, image_dir, pathologies):
    """Evaluate trained model on test set"""
    
    # Load model
    print("📊 Loading trained model...")
    model = load_trained_model(model_path, num_classes=len(pathologies))
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load test data
    test_dataset = ChestXRayDataset(test_csv, image_dir)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Collect predictions
    all_probs = []
    all_labels = []
    all_preds = []
    
    print("🔍 Running evaluation...")
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            
            probabilities, _ = model(images)
            predictions = (probabilities > 0.5).float()
            
            all_probs.append(probabilities.cpu().numpy())
            all_labels.append(labels.numpy())
            all_preds.append(predictions.cpu().numpy())
    
    # Concatenate results
    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)
    
    # Calculate metrics
    results = {}
    
    print("\n📈 Per-class Results:")
    print("-" * 60)
    print(f"{'Pathology':<20} {'AUC':<8} {'AP':<8} {'Sensitivity':<12} {'Specificity':<12}")
    print("-" * 60)
    
    for i, pathology in enumerate(pathologies):
        if len(np.unique(all_labels[:, i])) > 1:
            auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
            ap = average_precision_score(all_labels[:, i], all_probs[:, i])
            
            # Calculate sensitivity and specificity
            tp = np.sum((all_labels[:, i] == 1) & (all_preds[:, i] == 1))
            tn = np.sum((all_labels[:, i] == 0) & (all_preds[:, i] == 0))
            fp = np.sum((all_labels[:, i] == 0) & (all_preds[:, i] == 1))
            fn = np.sum((all_labels[:, i] == 1) & (all_preds[:, i] == 0))
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            results[pathology] = {
                'auc': auc,
                'ap': ap,
                'sensitivity': sensitivity,
                'specificity': specificity
            }
            
            print(f"{pathology:<20} {auc:<8.3f} {ap:<8.3f} {sensitivity:<12.3f} {specificity:<12.3f}")
        else:
            print(f"{pathology:<20} {'N/A':<8} {'N/A':<8} {'N/A':<12} {'N/A':<12}")
    
    # Overall metrics
    valid_aucs = [results[p]['auc'] for p in results if 'auc' in results[p]]
    mean_auc = np.mean(valid_aucs) if valid_aucs else 0
    
    print("-" * 60)
    print(f"{'Mean AUC':<20} {mean_auc:<8.3f}")
    print("-" * 60)
    
    return results, all_probs, all_labels, all_preds

def plot_results(results, pathologies, save_dir="results"):
    """Plot evaluation results"""
    os.makedirs(save_dir, exist_ok=True)
    
    # AUC scores plot
    aucs = [results.get(p, {}).get('auc', 0) for p in pathologies]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(pathologies)), aucs)
    plt.xlabel('Pathologies')
    plt.ylabel('AUC Score')
    plt.title('Per-Class AUC Scores')
    plt.xticks(range(len(pathologies)), pathologies, rotation=45, ha='right')
    plt.ylim(0, 1)
    
    # Color bars based on performance
    for i, bar in enumerate(bars):
        if aucs[i] > 0.8:
            bar.set_color('green')
        elif aucs[i] > 0.7:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'auc_scores.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Results saved to {save_dir}/")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model', type=str, default='models/chest_xray_classifier.pth',
                       help='Path to trained model')
    parser.add_argument('--test_csv', type=str, default='data/processed/test_chestx14.csv',
                       help='Test dataset CSV')
    parser.add_argument('--image_dir', type=str, default='data/raw/chestx_ray14/images',
                       help='Image directory')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Pathologies
    pathologies = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
        "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
        "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
    ]
    
    print("🏥 Model Evaluation")
    print("=" * 40)
    
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        print("   Train a model first using scripts/train_classifier.py")
        return 1
    
    if not os.path.exists(args.test_csv):
        print(f"❌ Test CSV not found: {args.test_csv}")
        print("   Prepare test data first")
        return 1
    
    # Run evaluation
    results, probs, labels, preds = evaluate_model(
        args.model, args.test_csv, args.image_dir, pathologies
    )
    
    # Plot results
    plot_results(results, pathologies, args.output_dir)
    
    # Save detailed results
    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(args.output_dir, 'detailed_results.csv'))
    
    print(f"\n✅ Evaluation completed! Results saved to {args.output_dir}/")
    
    return 0

if __name__ == "__main__":
    exit(main())