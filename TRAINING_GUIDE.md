# Training Guide - Individual Models

This guide helps you train models **one at a time** to avoid memory/resource issues on laptops with limited resources.

## 📊 Data Split Configuration

All models now use: **70% Training / 30% Testing**

- CNN and ResNet additionally use 20% of training data for internal validation

## 🎯 Training Order (Recommended)

### Step 1: Train Random Forest (Fastest)

```bash
python models/random_forest/train.py
```

- **Time:** 10-30 minutes
- **Memory:** Low
- **Output:** `results/random_forest/`
  - `rf_model.pkl` - Trained model
  - `metrics.json` - Performance metrics
  - `confusion_matrix.png` - Confusion matrix
  - `per_class_metrics.png` - Per-class performance chart

---

### Step 2: Train XGBoost

```bash
python models/xgboost/train.py
```

- **Time:** 15-40 minutes
- **Memory:** Low-Medium
- **Output:** `results/xgboost/`
  - `xgb_model.pkl` - Trained model
  - `metrics.json` - Performance metrics
  - `confusion_matrix.png` - Confusion matrix
  - `per_class_metrics.png` - Per-class performance chart
  - `feature_importance.png` - Top features

---

### Step 3: Train CNN

```bash
python models/cnn/train.py
```

- **Time:** 1-2 hours
- **Memory:** Medium-High
- **GPU:** Recommended but optional
- **Output:** `results/cnn/`
  - `cnn_model.h5` - Trained model
  - `metrics.json` - Performance metrics
  - `confusion_matrix.png` - Confusion matrix
  - `per_class_metrics.png` - Per-class performance chart
  - `training_history.png` - Training curves

**💡 Tip:** Close other applications to free up RAM before training CNN.

---

### Step 4: Train ResNet

```bash
python models/resnet/train.py
```

- **Time:** 2-3 hours
- **Memory:** Medium-High
- **GPU:** Recommended but optional
- **Output:** `results/resnet/`
  - `resnet_model.h5` - Trained model
  - `metrics.json` - Performance metrics
  - `confusion_matrix.png` - Confusion matrix
  - `per_class_metrics.png` - Per-class performance chart
  - `training_history.png` - Training curves

**💡 Tip:** Train overnight if your laptop is slow.

---

## 🔍 After Training

### View Individual Model Results

Each model saves its results in `results/<model_name>/`:

- Check `metrics.json` for accuracy scores
- View confusion matrices as PNG images
- Review per-class performance charts

### Use Streamlit App

```bash
streamlit run app.py
```

The app will automatically load all trained models and allow you to:

- Upload files for prediction
- Compare predictions across all models
- View training metrics and confusion matrices
- Download comprehensive reports

### Compare All Models

After training all 4 models:

```bash
python compare_models.py
```

Generates:

- `results/model_comparison.png` - Overall performance comparison
- `results/per_class_comparison.png` - Per-class accuracy comparison
- `results/comparison_report.txt` - Detailed text report

---

## ⚙️ Training Configuration

### Modify Training Parameters

#### Random Forest

Edit `models/random_forest/train.py`:

```python
rf_classifier = RandomForestFileClassifier(
    n_estimators=100,  # Number of trees (increase for better accuracy)
    max_depth=20       # Maximum tree depth
)
```

#### XGBoost

Edit `models/xgboost/train.py`:

```python
xgb_classifier = XGBoostFileClassifier(
    n_estimators=100,    # Number of boosting rounds
    max_depth=6,         # Maximum tree depth
    learning_rate=0.1,   # Learning rate
    use_gpu=False        # Set True if you have NVIDIA GPU
)
```

#### CNN

Edit `models/cnn/train.py`:

```python
cnn_classifier.train(
    X_train, y_train, None, None, file_types,
    epochs=50,           # Number of training epochs
    batch_size=64,       # Batch size (reduce if memory issues)
    validation_split=0.2 # Validation percentage
)
```

#### ResNet

Edit `models/resnet/train.py`:

```python
resnet_classifier.train(
    X_train, y_train, None, None, file_types,
    epochs=50,           # Number of training epochs
    batch_size=64,       # Batch size (reduce if memory issues)
    validation_split=0.2 # Validation percentage
)
```

---

## 🚨 Troubleshooting

### Memory Error

**Problem:** Out of memory during training

**Solution:**

1. Close all other applications
2. For CNN/ResNet, reduce batch size:
   ```python
   batch_size=32  # or even 16
   ```
3. Train models one at a time (never run multiple simultaneously)

### Slow Training

**Problem:** Training takes too long

**Solution:**

1. For Random Forest: Reduce `n_estimators` or `max_depth`
2. For XGBoost: Reduce `n_estimators`
3. For CNN/ResNet:
   - Reduce `epochs` (try 30 instead of 50)
   - Let it train overnight
   - Consider using GPU if available

### Model Not Found in Streamlit

**Problem:** Streamlit app says "No models found"

**Solution:**

1. Ensure you've completed training at least one model
2. Check that `results/<model_name>/` folder exists
3. Verify model files exist (`.pkl` or `.h5`)

---

## 📈 Expected Performance

After training, expect these approximate accuracies:

- **Random Forest:** 50-70%
- **XGBoost:** 60-75%
- **CNN:** 70-85%
- **ResNet:** 75-90%

Actual performance depends on your data quality and training configuration.

---

## ✅ Checklist

- [ ] Train Random Forest
- [ ] Train XGBoost
- [ ] Train CNN
- [ ] Train ResNet
- [ ] Run Streamlit app
- [ ] Test with sample files
- [ ] Generate comparison report

Happy Training! 🚀
