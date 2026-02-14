"""
Enhanced training script for Random Forest with comprehensive per-class metrics.
"""

import os
import sys
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add parent directory to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from utils.data_loader import load_train_data
from config import RANDOM_SEED


class RandomForestFileClassifier:
    """Random Forest classifier with comprehensive metrics."""
    
    def __init__(self, n_estimators=100, max_depth=None, random_state=RANDOM_SEED):
        """Initialize Random Forest classifier."""
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )
        self.file_types = None
        self.training_time = None
        
    def train(self, X_train, y_train, file_types):
        """Train the model."""
        import time
        
        print("\n" + "="*80)
        print("TRAINING RANDOM FOREST MODEL")
        print("="*80)
        print(f"Training samples: {len(X_train):,}")
        print(f"Features per sample: {X_train.shape[1]:,}")
        print(f"Number of classes: {len(file_types)}")
        print()
        
        self.file_types = file_types
        
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start_time
        
        print(f"\n✅ Training completed in {self.training_time:.2f} seconds")
        
    def evaluate(self, X_test, y_test, save_dir=None):
        """
        Evaluate model with comprehensive per-class metrics.
        
        Args:
            X_test: Test features
            y_test: Test labels
            save_dir: Directory to save results
            
        Returns:
            Dictionary of metrics
        """
        print("\n" + "="*80)
        print("MODEL EVALUATION")
        print("="*80)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Overall metrics
        overall_accuracy = accuracy_score(y_test, y_pred)
        overall_precision = precision_score(y_test, y_pred, average='weighted')
        overall_recall = recall_score(y_test, y_pred, average='weighted')
        overall_f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\n📊 OVERALL METRICS:")
        print(f"  Accuracy:  {overall_accuracy*100:.2f}%")
        print(f"  Precision: {overall_precision*100:.2f}%")
        print(f"  Recall:    {overall_recall*100:.2f}%")
        print(f"  F1-Score:  {overall_f1*100:.2f}%")
        
        # Per-class metrics
        print(f"\n📋 PER-CLASS METRICS:")
        print(f"{'File Type':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 75)
        
        # Get classification report as dict
        report = classification_report(y_test, y_pred, target_names=self.file_types, 
                                      output_dict=True, zero_division=0)
        
        per_class_metrics = {}
        for file_type in self.file_types:
            if file_type in report:
                metrics = report[file_type]
                
                # Calculate per-class accuracy
                class_idx = self.file_types.index(file_type)
                class_mask = (y_test == class_idx)
                class_accuracy = accuracy_score(y_test[class_mask], y_pred[class_mask])
                
                print(f"{file_type:<15} {class_accuracy*100:>8.2f}%  "
                      f"{metrics['precision']*100:>8.2f}%  "
                      f"{metrics['recall']*100:>8.2f}%  "
                      f"{metrics['f1-score']*100:>8.2f}%  "
                      f"{int(metrics['support']):>8,}")
                
                per_class_metrics[file_type] = {
                    'accuracy': class_accuracy,
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1-score'],
                    'support': int(metrics['support'])
                }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Save results
        results = {
            'model': 'Random Forest',
            'overall': {
                'accuracy': overall_accuracy,
                'precision': overall_precision,
                'recall': overall_recall,
                'f1_score': overall_f1
            },
            'per_class': per_class_metrics,
            'training_time': self.training_time,
            'timestamp': datetime.now().isoformat()
        }
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save metrics as JSON
            with open(save_dir / 'metrics.json', 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Metrics saved to {save_dir / 'metrics.json'}")
            
            # Save confusion matrix plot
            self.plot_confusion_matrix(cm, save_dir / 'confusion_matrix.png')
            
            # Save feature importance
            self.plot_feature_importance(save_dir / 'feature_importance.png')
        
        return results
    
    def plot_confusion_matrix(self, cm, save_path):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.file_types,
                   yticklabels=self.file_types)
        plt.title('Confusion Matrix - Random Forest')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"📊 Confusion matrix saved to {save_path}")
    
    def plot_feature_importance(self, save_path, top_n=50):
        """Plot top N feature importances."""
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 8))
        plt.title(f'Top {top_n} Feature Importances - Random Forest')
        plt.bar(range(top_n), importances[indices])
        plt.xlabel('Byte Position')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"📊 Feature importance plot saved to {save_path}")
    
    def save_model(self, filepath):
        """Save trained model (sklearn model only)."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"💾 Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath):
        """Load trained model."""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"📂 Model loaded from {filepath}")
        return model


def main():
    """Main training pipeline."""
    from sklearn.model_selection import train_test_split
    
    print("="*80)
    print("RANDOM FOREST FILE TYPE CLASSIFIER")
    print("="*80)
    print()
    
    # Load data
    X, y, file_types, loader = load_train_data(
        train_dir=r"c:\Users\prath\Desktop\file-type-identification\Train",
        normalize=False,  # Random Forest doesn't need normalization
        verbose=True
    )
    
    # Split into train and test
    print("Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Train set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    print()
    
    # Create and train model
    rf_classifier = RandomForestFileClassifier(n_estimators=100, max_depth=20)
    rf_classifier.train(X_train, y_train, file_types)
    
    # Evaluate
    results = rf_classifier.evaluate(X_test, y_test, 
                                     save_dir='results/random_forest')
    
    # Save model
    rf_classifier.save_model('results/random_forest/rf_model.pkl')
    
    # Save label encoder
    loader.save_label_encoder('results/label_encoder.pkl')
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
