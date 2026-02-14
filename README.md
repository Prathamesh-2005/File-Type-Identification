# Intelligent File Type Classification System

## Problem Statement

Design and development of an intelligent file type classification system for fragmented and headerless file segments in data carving using deep learning and byte-pattern analysis.

## ✅ Project Status: READY TO TRAIN

- **✅ Data Verified**: 315,392 headerless fragments (4,096 bytes each)
- **✅ Models Ready**: 4 complete training scripts with comprehensive per-class metrics
- **✅ File Types**: 12 types (7zip, apk, bin, css, elf, html, javascript, json, mp3, mp4, pdf, tif)

## 🚀 Quick Start

### Train All Models

```bash
python train_all_models.py
```

### Train Individual Models

```bash
# Random Forest
python models/random_forest/train.py

# XGBoost
python models/xgboost/train.py

# CNN
python models/cnn/train.py

# ResNet
python models/resnet/train.py
```

### Compare Results

```bash
python compare_models.py
```

All results are saved to the `results/` folder.

## Overview

This project implements a modular machine learning system for classifying file types from fragmented and headerless file segments. It addresses critical challenges in digital forensics and data carving where traditional header-footer based tools fail.

## Models

1. **Random Forest** - Tree-based ensemble (10-30 min training)
2. **XGBoost** - Gradient boosting (15-40 min training)
3. **CNN** - Convolutional neural network (1-2 hours training)
4. **ResNet** - Residual network with skip connections (2-3 hours training)

Each model outputs:

- Overall metrics (accuracy, precision, recall, F1-score)
- Per-class metrics for all 12 file types
- Confusion matrix visualization
- Feature importance / training history plots

## Dataset Distribution

| File Type  | Fragments   | Percentage |
| ---------- | ----------- | ---------- |
| 7zip       | 110,544     | 35.1%      |
| apk        | 53,432      | 16.9%      |
| tif        | 43,941      | 13.9%      |
| mp4        | 38,866      | 12.3%      |
| pdf        | 26,045      | 8.3%       |
| mp3        | 20,937      | 6.6%       |
| bin        | 14,867      | 4.7%       |
| json       | 3,161       | 1.0%       |
| javascript | 1,280       | 0.4%       |
| elf        | 1,217       | 0.4%       |
| css        | 647         | 0.2%       |
| html       | 455         | 0.1%       |
| **Total**  | **315,392** | **100%**   |

## Project Structure

```
file-type-identification/
├── config.py                       # Global configuration
├── train_all_models.py             # Master training script
├── compare_models.py               # Model comparison tool
├── analyze_fragments.py            # Data analysis utility
├── Train/                          # Training data (315k fragments)
│   ├── 7zipFragments/
│   ├── apkFragments/
│   ├── pdfFragments/
│   └── ... (12 file types)
├── models/                         # Model implementations
│   ├── random_forest/
│   │   ├── __init__.py
│   │   └── train.py               # Random Forest training
│   ├── xgboost/
│   │   ├── __init__.py
│   │   └── train.py               # XGBoost training
│   ├── cnn/
│   │   ├── __init__.py
│   │   └── train.py               # CNN training
│   └── resnet/
│       ├── __init__.py
│       └── train.py               # ResNet training
├── results/                        # All training results
│   ├── random_forest/
│   │   ├── metrics.json
│   │   ├── confusion_matrix.png
│   │   ├── feature_importance.png
│   │   └── rf_model.pkl
│   ├── xgboost/
│   ├── cnn/
│   ├── resnet/
│   ├── comparisons/                # Model comparison charts
│   ├── all_models_comparison.json
│   └── label_encoder.pkl
├── utils/                          # Utility functions
│   ├── __init__.py
│   ├── data_loader.py             # Fragment loading
│   └── header_footer_detector.py  # Header/footer detection
└── scripts/                        # Helper scripts
    ├── generate_fragments.py
    └── split_dataset.py
```

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
pip install tensorflow xgboost
```

## Usage

### 1. Prepare Your Data

Place your original files in the `data/raw` directory, organized by file type:

```
data/raw/
├── pdf/
│   ├── document1.pdf
│   └── document2.pdf
├── jpg/
│   ├── image1.jpg
│   └── image2.jpg
└── mp3/
    ├── audio1.mp3
    └── audio2.mp3
```

### 2. Generate Fragments

Generate fragments with automatic header/footer detection:

```bash
python scripts/generate_fragments.py --input-dir data/raw --output-dir data/fragments --remove-headers --remove-footers
```

Generate fragments with forced header/footer removal:

```bash
python scripts/generate_fragments.py --input-dir data/raw --output-dir data/fragments --force-header-bytes 64 --force-footer-bytes 64
```

Generate fragments for specific file types:

```bash
python scripts/generate_fragments.py --input-dir data/raw --output-dir data/fragments --extensions pdf jpg mp3
```

**Options:**

- `--fragment-size`: Size of each fragment in bytes (default: 1024)
- `--remove-headers`: Remove file headers using signature detection
- `--remove-footers`: Remove file footers using signature detection
- `--force-header-bytes N`: Force removal of N bytes from start
- `--force-footer-bytes N`: Force removal of N bytes from end
- `--extensions`: List of file extensions to process

### 3. Split Dataset

Split fragments into training (70%), testing (15%), and validation (15%) sets:

```bash
python scripts/split_dataset.py --fragments-dir data/fragments --train-dir data/train --test-dir data/test --validation-dir data/validation
```

**Options:**

- `--train-ratio`: Training set ratio (default: 0.70)
- `--test-ratio`: Test set ratio (default: 0.15)
- `--validation-ratio`: Validation set ratio (default: 0.15)
- `--random-seed`: Random seed for reproducibility (default: 42)

### 4. Train Models

#### Random Forest

```bash
python models/random_forest/train.py
```

#### CNN

```bash
python models/cnn/train.py
```

#### XGBoost

```bash
python models/xgboost/train.py
```

#### ResNet

```bash
python models/resnet/train.py
```

Each training script will:

- Load the training, validation, and test data
- Train the model with the specified configuration
- Evaluate on validation and test sets
- Generate visualizations (confusion matrices, training curves)
- Save the trained model

### 5. Analyze Files

Analyze a file for header/footer presence:

```bash
python utils/header_footer_detector.py path/to/file.pdf
```

## Configuration

Edit `config.py` to customize:

- **Fragment size**: `FRAGMENT_SIZE = 1024` (bytes)
- **Data split ratios**: `TRAIN_RATIO`, `TEST_RATIO`, `VALIDATION_RATIO`
- **Model hyperparameters**:
  - Random Forest: `RF_N_ESTIMATORS`, `RF_MAX_DEPTH`
  - CNN: `CNN_EPOCHS`, `CNN_BATCH_SIZE`, `CNN_LEARNING_RATE`
  - XGBoost: `XGB_N_ESTIMATORS`, `XGB_MAX_DEPTH`, `XGB_LEARNING_RATE`
  - ResNet: `RESNET_EPOCHS`, `RESNET_BATCH_SIZE`, `RESNET_LEARNING_RATE`
- **File signatures**: Add/modify signatures in `FILE_SIGNATURES` dictionary

## Supported File Types

Currently supported file types with header/footer detection:

- **Documents**: PDF, DOC, DOCX, XLSX, TXT
- **Images**: JPEG, PNG, GIF, BMP
- **Audio**: MP3, WAV
- **Video**: MP4, AVI
- **Archives**: ZIP
- **Executables**: EXE, ELF

## Model Descriptions

### Random Forest

- **Type**: Traditional machine learning
- **Approach**: Ensemble of decision trees on byte values
- **Best for**: Fast inference, feature importance analysis
- **Output**: Saved model in `models/random_forest/`

### CNN (Convolutional Neural Network)

- **Type**: Deep learning
- **Approach**: 1D convolutions to learn byte patterns
- **Architecture**: 4 conv blocks with batch normalization and dropout
- **Best for**: Spatial pattern recognition in byte sequences
- **Output**: Saved model in `models/cnn/`

### XGBoost

- **Type**: Gradient boosting
- **Approach**: Iterative boosting on decision trees
- **Best for**: High accuracy, handles imbalanced data well
- **Output**: Saved model in `models/xgboost/`

### ResNet (Residual Network)

- **Type**: Deep learning
- **Approach**: Residual connections for deep architecture
- **Architecture**: 4 stages with residual blocks
- **Best for**: Learning complex hierarchical patterns
- **Output**: Saved model in `models/resnet/`

## Literature Survey Context

### Gap Analysis

1. **Traditional Tools Limitations**:
   - Foremost, Scalpel, PhotoRec rely on header-footer matching
   - Fail when headers/footers are missing or corrupted
   - Lack context-awareness for fragmented files

2. **ML-Based Approaches**:
   - Existing solutions are format-specific (e.g., JPEG, PDF only)
   - Limited generalization to new or proprietary formats
   - Cannot handle fileless malware remnants

3. **Our Contribution**:
   - Multi-model approach for comparative analysis
   - Handles headerless and fragmented files
   - Extensible to new file types
   - Byte-pattern learning without relying on signatures

## Evaluation Metrics

Each model generates:

- **Accuracy scores** on validation and test sets
- **Classification reports** (precision, recall, F1-score per class)
- **Confusion matrices** (visualized as heatmaps)
- **Training curves** (for deep learning models)
- **Feature importance** (for tree-based models)

## Output Files

After training, each model directory contains:

- `*_model.pkl` or `*_model.h5`: Trained model weights
- `label_encoder.pkl`: Label encoding mapping
- `confusion_matrix_val.png`: Validation confusion matrix
- `confusion_matrix_test.png`: Test confusion matrix
- `training_history.png`: Training/validation curves (deep learning)
- `feature_importance.png`: Feature importance (tree-based models)

## Best Practices

1. **Data Preparation**:
   - Ensure balanced representation of file types
   - Use realistic file samples from target domain
   - Consider different file sizes and sources

2. **Fragment Generation**:
   - Test both with and without header/footer removal
   - Experiment with different fragment sizes
   - Validate that fragments maintain class diversity

3. **Model Training**:
   - Start with Random Forest for baseline
   - Use GPU for deep learning models (CNN, ResNet)
   - Monitor validation metrics to prevent overfitting
   - Adjust batch sizes based on available memory

4. **Evaluation**:
   - Always evaluate on truly held-out test data
   - Check confusion matrix for systematic misclassifications
   - Analyze difficult cases (low confidence predictions)

## Troubleshooting

### Memory Issues

- Reduce `FRAGMENT_SIZE` in `config.py`
- Decrease batch size for deep learning models
- Process fewer files at a time

### Poor Performance

- Increase training data
- Try different fragment sizes
- Adjust model hyperparameters
- Ensure data quality and balance

### Missing Dependencies

```bash
pip install --upgrade numpy pandas scikit-learn matplotlib seaborn tensorflow xgboost
```

## Future Enhancements

- [ ] Real-time classification API
- [ ] Ensemble model combining multiple classifiers
- [ ] Support for more file types
- [ ] Noise injection for robustness testing
- [ ] Transfer learning from pre-trained models
- [ ] Attention mechanisms for important byte positions
- [ ] Multi-fragment context aggregation

## Citation

If you use this system in your research, please cite:

```
@misc{file-type-classification,
  title={Intelligent File Type Classification System for Fragmented and Headerless File Segments},
  author={Your Name},
  year={2026},
  note={Deep Learning and Byte-Pattern Analysis for Data Carving}
}
```

## License

[Specify your license here]

## Contact

[Your contact information]

## Acknowledgments

This project addresses key challenges in digital forensics by combining traditional machine learning with deep learning approaches for robust file type classification in challenging scenarios.
