"""
Streamlit UI for File Type Prediction
"""

import streamlit as st
import pickle
import numpy as np
from pathlib import Path
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Page configuration
st.set_page_config(
    page_title="File Type Classifier",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def load_model_and_encoder():
    """Load trained model and label encoder (cached)."""
    # Try to find available models
    model_paths = {
        'Random Forest': Path('results/random_forest/rf_model.pkl'),
        'XGBoost': Path('results/xgboost/xgb_model.pkl'),
        'CNN': Path('results/cnn/cnn_model.h5'),
        'ResNet': Path('results/resnet/resnet_model.h5')
    }
    
    encoder_path = Path('results/label_encoder.pkl')
    
    # Find first available model
    model_name = None
    model_path = None
    for name, path in model_paths.items():
        if path.exists():
            model_name = name
            model_path = path
            break
    
    if model_path is None:
        return None, None, None, "No trained model found. Please train a model first."
    
    if not encoder_path.exists():
        return None, None, None, "Label encoder not found."
    
    try:
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load label encoder
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        
        return model, label_encoder, model_name, None
    except Exception as e:
        return None, None, None, f"Error loading model: {str(e)}"

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

def predict_file_type(file_data, model, label_encoder):
    """Predict file type from file data."""
    try:
        # Reshape for prediction
        file_data = file_data.reshape(1, -1)
        
        # Get prediction
        prediction = model.predict(file_data)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        
        # Get prediction probabilities
        proba = model.predict_proba(file_data)[0]
        confidence = np.max(proba) * 100
        
        # Get top 5 predictions
        top_indices = np.argsort(proba)[::-1][:5]
        top_predictions = []
        for idx in top_indices:
            file_type = label_encoder.inverse_transform([idx])[0]
            prob = proba[idx] * 100
            top_predictions.append((file_type, prob))
        
        return predicted_label, confidence, top_predictions
    
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None, None

def format_bytes(size):
    """Format bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"

def main():
    # Header
    st.title("🔍 File Type Classifier")
    st.markdown("### Intelligent file type identification using Machine Learning")
    st.markdown("---")
    
    # Load model
    with st.spinner("Loading AI model..."):
        model, label_encoder, model_name, error = load_model_and_encoder()
    
    if error:
        st.error(f"❌ {error}")
        st.info("Please train a model first by running:")
        st.code("python models/random_forest/train.py\n# or\npython models/xgboost/train.py")
        return
    
    st.success(f"✅ {model_name} Model loaded successfully!")
    
    # Sidebar info
    with st.sidebar:
        st.header("ℹ️ About")
        if model_name:
            st.info(f"**Active Model:** {model_name}")
        st.markdown("""
        This tool uses **Machine Learning** to identify file types from 
        binary content, even when headers are removed or corrupted.
        """)
        
        st.markdown("---")
        st.subheader("📋 Supported File Types")
        supported_types = [
            "7zip", "APK", "BIN", "CSS", "ELF", 
            "HTML", "JavaScript", "JSON", "MP3", 
            "MP4", "PDF", "TIF"
        ]
        for ft in supported_types:
            st.markdown(f"• {ft}")
        
        st.markdown("---")
        st.subheader("🎯 How it works")
        st.markdown("""
        1. Upload any file
        2. AI analyzes byte patterns
        3. Predicts file type with confidence
        4. Shows top 5 most likely types
        """)
        
        st.markdown("---")
        st.caption("Trained on 315,392 fragments")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📤 Upload File")
        uploaded_file = st.file_uploader(
            "Choose a file to analyze",
            type=None,
            help="Upload any file to predict its type"
        )
    
    with col2:
        if uploaded_file is not None:
            st.subheader("📊 File Info")
            file_size = uploaded_file.size
            st.metric("File Name", uploaded_file.name)
            st.metric("File Size", format_bytes(file_size))
    
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
        if st.button("🔮 Analyze File", type="primary", use_container_width=True):
            with st.spinner("Analyzing file..."):
                # Read fragment
                fragment = read_file_fragment(uploaded_file, offset=offset)
                
                if fragment is not None:
                    # Make prediction
                    predicted_type, confidence, top_predictions = predict_file_type(
                        fragment, model, label_encoder
                    )
                    
                    if predicted_type is not None:
                        st.markdown("---")
                        
                        # Results
                        st.subheader("🎯 Prediction Results")
                        
                        # Main prediction with big display
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(f"### Predicted File Type")
                            st.markdown(f"# 📄 **{predicted_type.upper()}**")
                        
                        with col2:
                            st.markdown(f"### Confidence")
                            st.markdown(f"# **{confidence:.1f}%**")
                            
                            # Confidence indicator
                            if confidence >= 80:
                                st.success("High confidence ✅")
                            elif confidence >= 60:
                                st.warning("Medium confidence ⚠️")
                            else:
                                st.error("Low confidence ❌")
                        
                        st.markdown("---")
                        
                        # Top predictions
                        st.subheader("📊 Top 5 Predictions")
                        
                        for i, (file_type, prob) in enumerate(top_predictions, 1):
                            # Create progress bar
                            col1, col2, col3 = st.columns([1, 4, 1])
                            
                            with col1:
                                st.markdown(f"**#{i}**")
                            
                            with col2:
                                st.progress(prob / 100)
                                st.caption(file_type.upper())
                            
                            with col3:
                                st.markdown(f"**{prob:.2f}%**")
                        
                        st.markdown("---")
                        
                        # Additional info
                        with st.expander("📈 Technical Details"):
                            st.json({
                                "predicted_type": predicted_type,
                                "confidence": f"{confidence:.2f}%",
                                "fragment_size": "4096 bytes",
                                "offset": offset,
                                "model": model_name,
                                "top_predictions": [
                                    {
                                        "rank": i,
                                        "type": ft,
                                        "probability": f"{p:.2f}%"
                                    }
                                    for i, (ft, p) in enumerate(top_predictions, 1)
                                ]
                            })
                        
                        # Download prediction results
                        st.download_button(
                            label="📥 Download Results (JSON)",
                            data=str({
                                "file_name": uploaded_file.name,
                                "predicted_type": predicted_type,
                                "confidence": confidence,
                                "top_predictions": top_predictions
                            }),
                            file_name=f"prediction_{uploaded_file.name}.json",
                            mime="application/json"
                        )

if __name__ == '__main__':
    main()
