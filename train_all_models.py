"""
Master training script to train all models and compare results.
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

print("=" * 80)
print("MASTER TRAINING SCRIPT - ALL MODELS")
print("=" * 80)
print()

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("📦 Loading dependencies...")
from utils.data_loader import load_train_data
from config import RANDOM_SEED
from sklearn.model_selection import train_test_split
import numpy as np

print("✅ Dependencies loaded\n")

# Track overall progress
overall_start_time = time.time()
all_results = {}

# ==================== LOAD DATA ====================
print("=" * 80)
print("STEP 1: LOADING DATA")
print("=" * 80)
print()

X, y, file_types, loader = load_train_data(
    train_dir=r"c:\Users\prath\Desktop\file-type-identification\Train",
    normalize=False,  # We'll normalize per model as needed
    verbose=True
)

print(f"\n✅ Data loaded successfully: {len(X):,} samples, {len(file_types)} classes")
print()

# ==================== SPLIT DATA ====================
print("=" * 80)
print("STEP 2: SPLITTING DATA")
print("=" * 80)
print()

# First split: 80% train+val, 20% test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

# Second split: 87.5% train, 12.5% val (of the 80%)
# This gives us 70% train, 10% val, 20% test overall
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.125, random_state=RANDOM_SEED, stratify=y_train_val
)

print(f"Train set:      {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"Validation set: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test set:       {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
print()

# Save label encoder (shared across all models)
Path('results').mkdir(exist_ok=True)
loader.save_label_encoder('results/label_encoder.pkl')
print()

# ==================== MODEL 1: RANDOM FOREST ====================
print("\n" + "=" * 80)
print("MODEL 1/4: RANDOM FOREST")
print("=" * 80)

try:
    sys.path.append('models/random_forest')
    from models.random_forest.train import RandomForestFileClassifier
    
    rf_start = time.time()
    
    # Random Forest doesn't need normalization
    rf_classifier = RandomForestFileClassifier(n_estimators=100, max_depth=20)
    rf_classifier.train(X_train, y_train, file_types)
    
    # Evaluate
    rf_results = rf_classifier.evaluate(X_test, y_test, 
                                       save_dir='results/random_forest')
    
    # Save model
    rf_classifier.save_model('results/random_forest/rf_model.pkl')
    
    rf_time = time.time() - rf_start
    all_results['Random Forest'] = {
        'metrics': rf_results,
        'total_time': rf_time
    }
    
    print(f"\n✅ Random Forest completed in {rf_time:.2f} seconds")
    
except Exception as e:
    print(f"\n❌ Random Forest failed: {e}")
    import traceback
    traceback.print_exc()

# ==================== MODEL 2: XGBOOST ====================
print("\n" + "=" * 80)
print("MODEL 2/4: XGBOOST")
print("=" * 80)

try:
    sys.path.append('models/xgboost')
    from models.xgboost.train import XGBoostFileClassifier
    
    xgb_start = time.time()
    
    # XGBoost doesn't need normalization
    xgb_classifier = XGBoostFileClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        use_gpu=False
    )
    xgb_classifier.train(X_train, y_train, file_types)
    
    # Evaluate
    xgb_results = xgb_classifier.evaluate(X_test, y_test, 
                                         save_dir='results/xgboost')
    
    # Save model
    xgb_classifier.save_model('results/xgboost/xgb_model.pkl')
    
    xgb_time = time.time() - xgb_start
    all_results['XGBoost'] = {
        'metrics': xgb_results,
        'total_time': xgb_time
    }
    
    print(f"\n✅ XGBoost completed in {xgb_time:.2f} seconds")
    
except Exception as e:
    print(f"\n❌ XGBoost failed: {e}")
    import traceback
    traceback.print_exc()

# ==================== MODEL 3: CNN ====================
print("\n" + "=" * 80)
print("MODEL 3/4: CNN")
print("=" * 80)

try:
    sys.path.append('models/cnn')
    from models.cnn.train import CNNFileClassifier
    
    cnn_start = time.time()
    
    # Normalize for CNN
    X_train_cnn = X_train.astype(np.float32) / 255.0
    X_val_cnn = X_val.astype(np.float32) / 255.0
    X_test_cnn = X_test.astype(np.float32) / 255.0
    
    cnn_classifier = CNNFileClassifier(
        input_shape=(X_train.shape[1], 1),
        num_classes=len(file_types)
    )
    cnn_classifier.train(X_train_cnn, y_train, X_val_cnn, y_val, file_types, 
                        epochs=50, batch_size=64)
    
    # Evaluate
    cnn_results = cnn_classifier.evaluate(X_test_cnn, y_test, 
                                         save_dir='results/cnn')
    
    # Save model
    cnn_classifier.save_model('results/cnn/cnn_model.h5')
    
    cnn_time = time.time() - cnn_start
    all_results['CNN'] = {
        'metrics': cnn_results,
        'total_time': cnn_time
    }
    
    print(f"\n✅ CNN completed in {cnn_time:.2f} seconds")
    
except Exception as e:
    print(f"\n❌ CNN failed: {e}")
    import traceback
    traceback.print_exc()

# ==================== MODEL 4: RESNET ====================
print("\n" + "=" * 80)
print("MODEL 4/4: RESNET")
print("=" * 80)

try:
    sys.path.append('models/resnet')
    from models.resnet.train import ResNetFileClassifier
    
    resnet_start = time.time()
    
    # Normalize for ResNet (reuse CNN normalization)
    X_train_resnet = X_train.astype(np.float32) / 255.0
    X_val_resnet = X_val.astype(np.float32) / 255.0
    X_test_resnet = X_test.astype(np.float32) / 255.0
    
    resnet_classifier = ResNetFileClassifier(
        input_shape=(X_train.shape[1], 1),
        num_classes=len(file_types)
    )
    resnet_classifier.train(X_train_resnet, y_train, X_val_resnet, y_val, file_types, 
                           epochs=50, batch_size=64)
    
    # Evaluate
    resnet_results = resnet_classifier.evaluate(X_test_resnet, y_test, 
                                               save_dir='results/resnet')
    
    # Save model
    resnet_classifier.save_model('results/resnet/resnet_model.h5')
    
    resnet_time = time.time() - resnet_start
    all_results['ResNet'] = {
        'metrics': resnet_results,
        'total_time': resnet_time
    }
    
    print(f"\n✅ ResNet completed in {resnet_time:.2f} seconds")
    
except Exception as e:
    print(f"\n❌ ResNet failed: {e}")
    import traceback
    traceback.print_exc()

# ==================== FINAL COMPARISON ====================
overall_time = time.time() - overall_start_time

print("\n\n" + "=" * 80)
print("FINAL RESULTS - MODEL COMPARISON")
print("=" * 80)
print()

if all_results:
    print(f"{'Model':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Time (s)':<12}")
    print("-" * 85)
    
    for model_name, data in sorted(all_results.items()):
        metrics = data['metrics']['overall']
        print(f"{model_name:<15} "
              f"{metrics['accuracy']*100:>10.2f}%  "
              f"{metrics['precision']*100:>10.2f}%  "
              f"{metrics['recall']*100:>10.2f}%  "
              f"{metrics['f1_score']*100:>10.2f}%  "
              f"{data['total_time']:>10.2f}")
    
    print()
    print("-" * 85)
    
    # Find best model
    best_model = max(all_results.items(), 
                    key=lambda x: x[1]['metrics']['overall']['accuracy'])
    print(f"🏆 BEST MODEL: {best_model[0]} "
          f"(Accuracy: {best_model[1]['metrics']['overall']['accuracy']*100:.2f}%)")
    
    # Save comprehensive results
    comprehensive_results = {
        'timestamp': datetime.now().isoformat(),
        'total_samples': len(X),
        'num_classes': len(file_types),
        'file_types': file_types,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'models': all_results,
        'best_model': best_model[0],
        'total_training_time': overall_time
    }
    
    # Save to JSON
    with open('results/all_models_comparison.json', 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    print(f"\n💾 Comprehensive results saved to results/all_models_comparison.json")

else:
    print("❌ No models were successfully trained")

print()
print("=" * 80)
print(f"✅ ALL TRAINING COMPLETED IN {overall_time:.2f} SECONDS")
print("=" * 80)
