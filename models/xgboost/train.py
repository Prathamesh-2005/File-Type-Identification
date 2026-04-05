"""
Enhanced training script for XGBoost with comprehensive per-class metrics.
"""

import os
import sys
import numpy as np
import pickle
import json
from pathlib import Path
import xgboost as xgb
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


class XGBoostFileClassifier:
    """XGBoost classifier with comprehensive metrics."""
    
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, 
                 random_state=RANDOM_SEED, use_gpu=False):
        """Initialize XGBoost classifier."""
        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'random_state': random_state,
            'objective': 'multi:softmax',
            'eval_metric': 'mlogloss',
            'tree_method': 'gpu_hist' if use_gpu else 'hist',
            'verbosity': 1,
            'subsample': 0.8,  # Use 80% of data per tree for memory efficiency
            'colsample_bytree': 0.8  # Use 80% of features per tree
        }
        
        self.model = xgb.XGBClassifier(**params)
        self.file_types = None
        self.training_time = None
        
    def train(self, X_train, y_train, file_types):
        """Train the model."""
        import time
        
        print("\n" + "="*80)
        print("TRAINING XGBOOST MODEL")
        print("="*80)
        print(f"Training samples: {len(X_train):,}")
        print(f"Features per sample: {X_train.shape[1]:,}")
        print(f"Number of classes: {len(file_types)}")
        print()
        
        self.file_types = file_types
        self.model.set_params(num_class=len(file_types))
        
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
            'model': 'XGBoost',
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
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                   xticklabels=self.file_types,
                   yticklabels=self.file_types)
        plt.title('Confusion Matrix - XGBoost')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"📊 Confusion matrix saved to {save_path}")
    
    def plot_feature_importance(self, save_path, top_n=50):
        """Plot top N feature importances."""
        importance_dict = self.model.get_booster().get_score(importance_type='weight')
        
        if importance_dict:
            # Get top N features
            sorted_importance = sorted(importance_dict.items(), 
                                     key=lambda x: x[1], reverse=True)[:top_n]
            features, importances = zip(*sorted_importance)
            
            plt.figure(figsize=(10, 8))
            plt.title(f'Top {top_n} Feature Importances - XGBoost')
            plt.bar(range(len(features)), importances)
            plt.xlabel('Feature Index')
            plt.ylabel('Importance (Weight)')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"📊 Feature importance plot saved to {save_path}")
    
    def save_model(self, filepath):
        """Save trained model."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
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
    print("XGBOOST FILE TYPE CLASSIFIER")
    print("="*80)
    print()
    
    # Load data
    X, y, file_types, loader = load_train_data(
        train_dir=r"c:\Users\prath\Desktop\file-type-identification\Train",
        normalize=False,  # XGBoost doesn't need normalization
        verbose=True
    )
    
    # Split into train and test
    print("Splitting data (70% train, 30% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Train set: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Test set: {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
    print()
    
    # Ensure data stays as integer type (no float conversion for memory efficiency)
    print(f"Data type: {X_train.dtype} (memory efficient)")
    print(f"Memory usage: ~{X_train.nbytes / (1024**3):.2f} GB")
    
    # MEMORY FIX: Use subset of training data to fit in RAM
    # XGBoost also needs memory, so we reduce samples if needed
    max_samples = 200000  # Maximum samples that fit in most laptops
    if len(X_train) > max_samples:
        print(f"\n⚠️  Memory optimization: Using {max_samples:,} samples (random subset)")
        print(f"   Original: {len(X_train):,} samples")
        
        # Stratified sampling to maintain class distribution
        from sklearn.model_selection import train_test_split
        X_train_subset, _, y_train_subset, _ = train_test_split(
            X_train, y_train, 
            train_size=max_samples, 
            random_state=RANDOM_SEED, 
            stratify=y_train
        )
        X_train = X_train_subset
        y_train = y_train_subset
        
        print(f"   Subset:   {len(X_train):,} samples")
        print(f"   This will fit in ~3 GB RAM (instead of 6.69 GB)")
    print()
    
    # Create and train model with memory-efficient parameters
    xgb_classifier = XGBoostFileClassifier(
        n_estimators=50,    # Reduced from 100 to save memory
        max_depth=5,        # Reduced depth for memory efficiency
        learning_rate=0.1,
        use_gpu=False       # Set to True if you have GPU
    )
    xgb_classifier.train(X_train, y_train, file_types)
    
    # Evaluate
    results = xgb_classifier.evaluate(X_test, y_test, 
                                     save_dir='results/xgboost')
    
    # Save model
    xgb_classifier.save_model('results/xgboost/xgb_model.pkl')
    
    # Save label encoder
    loader.save_label_encoder('results/label_encoder.pkl')
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
