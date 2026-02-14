"""
Enhanced training script for CNN with comprehensive per-class metrics.
"""

import os
import sys
import numpy as np
import json
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
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

# Set random seeds
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


class CNNFileClassifier:
    """1D CNN classifier for file type identification."""
    
    def __init__(self, input_shape, num_classes):
        """
        Initialize CNN model.
        
        Args:
            input_shape: Tuple representing input shape (fragment_size, 1)
            num_classes: Number of file type classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
        self.file_types = None
        self.history = None
        
    def _build_model(self):
        """Build 1D CNN architecture."""
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # Conv Block 1
            layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.25),
            
            # Conv Block 2
            layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.25),
            
            # Conv Block 3
            layers.Conv1D(256, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            
            # Conv Block 4
            layers.Conv1D(512, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.4),
            
            # Dense layers
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, file_types, 
              epochs=50, batch_size=32):
        """Train the model."""
        print("\n" + "="*80)
        print("TRAINING CNN MODEL")
        print("="*80)
        print(f"Training samples: {len(X_train):,}")
        print(f"Validation samples: {len(X_val):,}")
        print(f"Input shape: {self.input_shape}")
        print(f"Number of classes: {self.num_classes}")
        print()
        
        self.file_types = file_types
        self.model.summary()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Reshape for 1D CNN (samples, timesteps, channels)
        X_train = X_train.reshape(-1, X_train.shape[1], 1)
        X_val = X_val.reshape(-1, X_val.shape[1], 1)
        
        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✅ Training completed")
        
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
        
        # Reshape for CNN
        X_test = X_test.reshape(-1, X_test.shape[1], 1)
        
        # Make predictions
        y_pred_probs = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
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
            'model': 'CNN',
            'overall': {
                'accuracy': overall_accuracy,
                'precision': overall_precision,
                'recall': overall_recall,
                'f1_score': overall_f1
            },
            'per_class': per_class_metrics,
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
            
            # Save training history
            if self.history:
                self.plot_training_history(save_dir / 'training_history.png')
        
        return results
    
    def plot_confusion_matrix(self, cm, save_path):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                   xticklabels=self.file_types,
                   yticklabels=self.file_types)
        plt.title('Confusion Matrix - CNN')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"📊 Confusion matrix saved to {save_path}")
    
    def plot_training_history(self, save_path):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        ax1.plot(self.history.history['accuracy'], label='Train')
        ax1.plot(self.history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss plot
        ax2.plot(self.history.history['loss'], label='Train')
        ax2.plot(self.history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"📊 Training history saved to {save_path}")
    
    def save_model(self, filepath):
        """Save trained model."""
        self.model.save(filepath)
        print(f"💾 Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath):
        """Load trained model."""
        model = keras.models.load_model(filepath)
        print(f"📂 Model loaded from {filepath}")
        return model


def main():
    """Main training pipeline."""
    from sklearn.model_selection import train_test_split
    
    print("="*80)
    print("CNN FILE TYPE CLASSIFIER")
    print("="*80)
    print()
    
    # Load data
    X, y, file_types, loader = load_train_data(
        train_dir=r"c:\Users\prath\Desktop\file-type-identification\Train",
        normalize=True,  # CNN needs normalization
        verbose=True
    )
    
    # Split into train, validation, and test
    print("Splitting data (70% train, 15% val, 15% test)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=y_temp
    )
    
    print(f"Train set: {len(X_train):,} samples")
    print(f"Validation set: {len(X_val):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    print()
    
    # Create and train model
    cnn_classifier = CNNFileClassifier(
        input_shape=(X_train.shape[1], 1),
        num_classes=len(file_types)
    )
    cnn_classifier.train(X_train, y_train, X_val, y_val, file_types, 
                        epochs=50, batch_size=64)
    
    # Evaluate
    results = cnn_classifier.evaluate(X_test, y_test, 
                                     save_dir='results/cnn')
    
    # Save model
    cnn_classifier.save_model('results/cnn/cnn_model.h5')
    
    # Save label encoder
    loader.save_label_encoder('results/label_encoder.pkl')
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
