"""
Streamlit UI for File Type Prediction - Multi-Model Comparison
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import sys
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import model classes for unpickling
try:
    from models.random_forest.train import RandomForestFileClassifier
    from models.xgboost.train import XGBoostFileClassifier
except ImportError:
    pass  # Classes not needed if models saved correctly

# Page configuration
st.set_page_config(
    page_title="Multi-Model File Type Classifier",
    page_icon="🔍",
    layout="wide"
)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

@st.cache_resource
def load_all_models():
    """Load all available trained models and label encoder."""
    model_configs = {
        'Random Forest': {
            'path': Path('results/random_forest/rf_model.pkl'),
            'metrics': Path('results/random_forest/metrics.json'),
            'confusion': Path('results/random_forest/confusion_matrix.png'),
            'type': 'sklearn'
        },
        'XGBoost': {
            'path': Path('results/xgboost/xgb_model.pkl'),
            'metrics': Path('results/xgboost/metrics.json'),
            'confusion': Path('results/xgboost/confusion_matrix.png'),
            'type': 'sklearn'
        },
        'CNN': {
            'path': Path('results/cnn/cnn_model.h5'),
            'metrics': Path('results/cnn/metrics.json'),
            'confusion': Path('results/cnn/confusion_matrix.png'),
            'type': 'keras'
        },
        'ResNet': {
            'path': Path('results/resnet/resnet_model.h5'),
            'metrics': Path('results/resnet/metrics.json'),
            'confusion': Path('results/resnet/confusion_matrix.png'),
            'type': 'keras'
        }
    }
    
    encoder_path = Path('results/label_encoder.pkl')
    
    if not encoder_path.exists():
        return None, None, "Label encoder not found. Please train models first."
    
    # Load label encoder
    try:
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
    except Exception as e:
        return None, None, f"Error loading label encoder: {str(e)}"
    
    # Load all available models
    loaded_models = {}
    for name, config in model_configs.items():
        if config['path'].exists():
            try:
                if config['type'] == 'keras':
                    # Lazy import TensorFlow
                    import tensorflow as tf
                    model = tf.keras.models.load_model(config['path'])
                else:
                    with open(config['path'], 'rb') as f:
                        loaded_obj = pickle.load(f)
                    
                    # Handle both wrapper classes and direct sklearn models
                    if hasattr(loaded_obj, 'model'):
                        # It's a wrapper class (XGBoostFileClassifier, etc.)
                        model = loaded_obj.model
                    else:
                        # It's a direct sklearn model (RandomForestClassifier, etc.)
                        model = loaded_obj
                
                # Load metrics if available
                metrics = None
                if config['metrics'].exists():
                    with open(config['metrics'], 'r') as f:
                        metrics = json.load(f)
                
                loaded_models[name] = {
                    'model': model,
                    'metrics': metrics,
                    'confusion_path': config['confusion'],
                    'type': config['type']
                }
            except Exception as e:
                st.warning(f"Could not load {name}: {str(e)}")
    
    if not loaded_models:
        return None, None, "No trained models found. Please train models first."
    
    return loaded_models, label_encoder, None

def read_file_fragment(uploaded_file, fragment_size=4096, offset=0):
    """Read a fragment from uploaded file."""
    try:
        # Reset file pointer
        uploaded_file.seek(0)
        
        # Seek to offset
        uploaded_file.seek(offset)
        
        # Read data
        data = uploaded_file.read(fragment_size)
        
        if len(data) < fragment_size:
            # Pad with zeros if file is smaller
            data = data + b'\x00' * (fragment_size - len(data))
        
        return np.frombuffer(data, dtype=np.uint8)
    
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

def predict_with_model(file_data, model_info, label_encoder, model_name):
    """Predict file type with a single model."""
    try:
        model = model_info['model']
        model_type = model_info['type']
        
        if model_type == 'keras':
            # CNN/ResNet prediction
            input_data = file_data.reshape(1, 4096, 1).astype('float32') / 255.0
            proba = model.predict(input_data, verbose=0)[0]
        else:
            # Random Forest/XGBoost prediction
            input_data = file_data.reshape(1, -1)
            proba = model.predict_proba(input_data)[0]
        
        # Get predicted class
        predicted_idx = np.argmax(proba)
        predicted_label = label_encoder.inverse_transform([predicted_idx])[0]
        confidence = proba[predicted_idx] * 100
        
        # Get top 5 predictions
        top_indices = np.argsort(proba)[::-1][:5]
        top_predictions = []
        for idx in top_indices:
            file_type = label_encoder.inverse_transform([idx])[0]
            prob = proba[idx] * 100
            top_predictions.append((file_type, prob))
        
        return {
            'predicted_type': predicted_label,
            'confidence': confidence,
            'top_predictions': top_predictions,
            'probabilities': proba
        }
    
    except Exception as e:
        st.error(f"Prediction error with {model_name}: {e}")
        return None

def predict_all_models(file_data, models_dict, label_encoder):
    """Predict file type using all available models."""
    results = {}
    for model_name, model_info in models_dict.items():
        result = predict_with_model(file_data, model_info, label_encoder, model_name)
        if result:
            results[model_name] = result
    return results

def format_bytes(size):
    """Format bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"

def display_model_comparison(predictions, models_dict):
    """Display comparison of all model predictions."""
    st.subheader("📊 Model Comparison")
    
    # Create comparison dataframe
    comparison_data = []
    for model_name, result in predictions.items():
        comparison_data.append({
            'Model': model_name,
            'Prediction': result['predicted_type'].upper(),
            'Confidence': f"{result['confidence']:.2f}%",
            'Confidence_Value': result['confidence']
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Display as table
    st.dataframe(
        df[['Model', 'Prediction', 'Confidence']],
        use_container_width=True,
        hide_index=True
    )
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 4))
    models = df['Model']
    confidences = df['Confidence_Value']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(models)]
    
    bars = ax.barh(models, confidences, color=colors)
    ax.set_xlabel('Confidence (%)', fontsize=12)
    ax.set_title('Model Confidence Comparison', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 100])
    
    # Add value labels
    for i, (bar, conf) in enumerate(zip(bars, confidences)):
        ax.text(conf + 1, i, f'{conf:.1f}%', va='center', fontweight='bold')
    
    st.pyplot(fig)
    plt.close()

def display_prediction_details(model_name, result):
    """Display detailed prediction results for a single model."""
    with st.expander(f"🔍 {model_name} - Detailed Results", expanded=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"#### Predicted: **{result['predicted_type'].upper()}**")
            st.markdown(f"**Confidence:** {result['confidence']:.2f}%")
            
            # Confidence indicator
            if result['confidence'] >= 80:
                st.success("✅ High confidence")
            elif result['confidence'] >= 60:
                st.warning("⚠️ Medium confidence")
            else:
                st.error("❌ Low confidence")
        
        with col2:
            # Display training accuracy if available
            st.markdown("#### Model Performance")
            st.caption("(From training)")
        
        # Top 5 predictions
        st.markdown("##### Top 5 Predictions:")
        for i, (file_type, prob) in enumerate(result['top_predictions'], 1):
            col1, col2, col3 = st.columns([0.5, 3, 1])
            
            with col1:
                st.markdown(f"**#{i}**")
            
            with col2:
                st.progress(prob / 100)
                st.caption(file_type.upper())
            
            with col3:
                st.markdown(f"**{prob:.2f}%**")

def display_model_training_stats(models_dict):
    """Display training statistics for all models."""
    st.subheader("📈 Model Training Performance")
    
    # Collect metrics
    metrics_data = []
    for model_name, model_info in models_dict.items():
        if model_info['metrics']:
            overall = model_info['metrics'].get('overall', {})
            metrics_data.append({
                'Model': model_name,
                'Accuracy': f"{overall.get('accuracy', 0) * 100:.2f}%",
                'Precision': f"{overall.get('precision', 0) * 100:.2f}%",
                'Recall': f"{overall.get('recall', 0) * 100:.2f}%",
                'F1-Score': f"{overall.get('f1_score', 0) * 100:.2f}%"
            })
    
    if metrics_data:
        df = pd.DataFrame(metrics_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.warning("Training metrics not available. Train models to see performance stats.")

def display_confusion_matrices(models_dict):
    """Display confusion matrices for all models."""
    st.subheader("🎯 Confusion Matrices")
    
    available_matrices = [(name, info['confusion_path']) 
                         for name, info in models_dict.items() 
                         if info['confusion_path'].exists()]
    
    if not available_matrices:
        st.warning("Confusion matrices not available. Train models to generate them.")
        return
    
    # Display in columns
    cols = st.columns(2)
    for idx, (model_name, confusion_path) in enumerate(available_matrices):
        with cols[idx % 2]:
            st.markdown(f"#### {model_name}")
            st.image(str(confusion_path), use_container_width=True)

def display_per_class_metrics(models_dict, label_encoder):
    """Display per-class metrics for all models."""
    st.subheader("📋 Per-Class Performance")
    
    # Get all file types
    file_types = label_encoder.classes_
    
    # Select file type to view
    selected_type = st.selectbox("Select file type:", file_types)
    
    # Collect per-class metrics for selected type
    metrics_data = []
    for model_name, model_info in models_dict.items():
        if model_info['metrics'] and 'per_class' in model_info['metrics']:
            class_metrics = model_info['metrics']['per_class'].get(selected_type, {})
            if class_metrics:
                metrics_data.append({
                    'Model': model_name,
                    'Accuracy': f"{class_metrics.get('accuracy', 0) * 100:.2f}%",
                    'Precision': f"{class_metrics.get('precision', 0) * 100:.2f}%",
                    'Recall': f"{class_metrics.get('recall', 0) * 100:.2f}%",
                    'F1-Score': f"{class_metrics.get('f1_score', 0) * 100:.2f}%",
                    'Support': class_metrics.get('support', 0)
                })
    
    if metrics_data:
        df = pd.DataFrame(metrics_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.warning(f"No metrics available for {selected_type}")

def main():
    # Header
    st.title("🔍 Multi-Model File Type Classifier")
    st.markdown("### Compare predictions across all trained models")
    st.markdown("---")
    
    # Load all models
    with st.spinner("Loading AI models..."):
        models_dict, label_encoder, error = load_all_models()
    
    if error:
        st.error(f"❌ {error}")
        st.info("Please train models first by running:")
        st.code("python train_all_models.py")
        return
    
    st.success(f"✅ {len(models_dict)} model(s) loaded successfully!")
    
    # Display loaded models
    model_names = ", ".join(models_dict.keys())
    st.info(f"**Loaded Models:** {model_names}")
    
    # Sidebar info
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown(f"""
        **Active Models:** {len(models_dict)}
        
        {chr(10).join([f"• {name}" for name in models_dict.keys()])}
        """)
        
        st.markdown("---")
        st.subheader("📋 Supported File Types")
        supported_types = label_encoder.classes_
        for ft in supported_types:
            st.markdown(f"• {ft}")
        
        st.markdown("---")
        st.subheader("🎯 Features")
        st.markdown("""
        ✅ Multi-model comparison
        ✅ Training metrics & accuracy
        ✅ Confusion matrices
        ✅ Per-class performance
        ✅ Comprehensive reports
        """)
        
        st.markdown("---")
        st.caption(f"Total file types: {len(supported_types)}")
    
    # Main content - File upload
    st.subheader("📤 Upload File for Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a file to analyze",
            type=None,
            help="Upload any file to predict its type with all models"
        )
    
    with col2:
        if uploaded_file is not None:
            st.metric("File Name", uploaded_file.name)
            st.metric("File Size", format_bytes(uploaded_file.size))
    
    # Analysis section
    if uploaded_file is not None:
        st.markdown("---")
        
        # Add offset selector for large files
        max_offset = max(0, uploaded_file.size - 4096)
        if max_offset > 0:
            offset = st.slider(
                "Fragment Offset (bytes)",
                min_value=0,
                max_value=max_offset,
                value=0,
                step=4096,
                help="Select which part of the file to analyze"
            )
        else:
            offset = 0
        
        # Predict button
        if st.button("🔮 Analyze with All Models", type="primary", use_container_width=True):
            with st.spinner("Analyzing file with all models..."):
                # Read fragment
                fragment = read_file_fragment(uploaded_file, offset=offset)
                
                if fragment is not None:
                    # Predict with all models
                    predictions = predict_all_models(fragment, models_dict, label_encoder)
                    
                    if predictions:
                        st.markdown("---")
                        
                        # Main results section
                        st.header("🎯 Prediction Results")
                        
                        # Model comparison
                        display_model_comparison(predictions, models_dict)
                        
                        st.markdown("---")
                        
                        # Detailed results for each model
                        st.header("🔍 Detailed Model Results")
                        for model_name, result in predictions.items():
                            display_prediction_details(model_name, result)
                        
                        st.markdown("---")
                        
                        # Training performance section
                        st.header("📊 Model Training Performance")
                        
                        # Overall metrics
                        display_model_training_stats(models_dict)
                        
                        st.markdown("---")
                        
                        # Confusion matrices
                        display_confusion_matrices(models_dict)
                        
                        st.markdown("---")
                        
                        # Per-class metrics
                        display_per_class_metrics(models_dict, label_encoder)
                        
                        st.markdown("---")
                        
                        # Download comprehensive report
                        st.subheader("📥 Download Report")
                        
                        report_data = {
                            "file_info": {
                                "name": uploaded_file.name,
                                "size": uploaded_file.size,
                                "offset": offset
                            },
                            "predictions": {}
                        }
                        
                        for model_name, result in predictions.items():
                            report_data["predictions"][model_name] = {
                                "predicted_type": result['predicted_type'],
                                "confidence": result['confidence'],
                                "top_5_predictions": [
                                    {"type": ft, "probability": float(prob)}
                                    for ft, prob in result['top_predictions']
                                ]
                            }
                        
                        # Add training metrics
                        report_data["training_metrics"] = {}
                        for model_name, model_info in models_dict.items():
                            if model_info['metrics']:
                                report_data["training_metrics"][model_name] = model_info['metrics']
                        
                        report_json = json.dumps(report_data, indent=2)
                        
                        st.download_button(
                            label="📥 Download Complete Report (JSON)",
                            data=report_json,
                            file_name=f"file_analysis_{uploaded_file.name}.json",
                            mime="application/json",
                            use_container_width=True
                        )
    
    # Model training stats (always visible)
    else:
        st.markdown("---")
        st.header("📊 Model Training Statistics")
        
        display_model_training_stats(models_dict)
        
        st.markdown("---")
        
        display_confusion_matrices(models_dict)

if __name__ == '__main__':
    main()
