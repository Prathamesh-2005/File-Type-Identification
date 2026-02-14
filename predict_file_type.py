"""
Predict file type from a random file using trained XGBoost model.
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_model_and_encoder():
    """Load trained model and label encoder."""
    model_path = Path('results/xgboost/xgb_model.pkl')
    encoder_path = Path('results/label_encoder.pkl')
    
    if not model_path.exists():
        print("❌ Error: Model not found at results/xgboost/xgb_model.pkl")
        print("   Please train the model first: python models/xgboost/train.py")
        return None, None
    
    if not encoder_path.exists():
        print("❌ Error: Label encoder not found at results/label_encoder.pkl")
        return None, None
    
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"✅ Model loaded from {model_path}")
    
    # Load label encoder
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    print(f"✅ Label encoder loaded from {encoder_path}")
    
    return model, label_encoder

def read_file_fragment(file_path, fragment_size=4096, offset=0):
    """
    Read a fragment from file.
    
    Args:
        file_path: Path to the file
        fragment_size: Size of fragment to read (default: 4096 bytes)
        offset: Byte offset to start reading from (default: 0)
        
    Returns:
        numpy array of bytes
    """
    try:
        with open(file_path, 'rb') as f:
            f.seek(offset)
            data = f.read(fragment_size)
            
        if len(data) < fragment_size:
            # Pad with zeros if file is smaller than fragment size
            data = data + b'\x00' * (fragment_size - len(data))
        
        return np.frombuffer(data, dtype=np.uint8)
    
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None

def predict_file_type(file_path, model, label_encoder):
    """
    Predict file type from a file.
    
    Args:
        file_path: Path to the file to analyze
        model: Trained classifier model
        label_encoder: Label encoder for converting predictions to file types
        
    Returns:
        Predicted file type
    """
    # Read file fragment
    fragment = read_file_fragment(file_path)
    if fragment is None:
        return None
    
    # Reshape for prediction
    fragment = fragment.reshape(1, -1)
    
    # Get prediction
    prediction = model.model.predict(fragment)
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    
    # Get prediction probabilities if available
    try:
        proba = model.model.predict_proba(fragment)[0]
        confidence = np.max(proba) * 100
        
        # Get top 3 predictions
        top_indices = np.argsort(proba)[::-1][:3]
        top_predictions = []
        for idx in top_indices:
            file_type = label_encoder.inverse_transform([idx])[0]
            prob = proba[idx] * 100
            top_predictions.append((file_type, prob))
        
        return predicted_label, confidence, top_predictions
    except:
        return predicted_label, None, None

def main():
    """Main prediction pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict file type from a file')
    parser.add_argument('file', type=str, help='Path to the file to analyze')
    parser.add_argument('--offset', type=int, default=0, 
                       help='Byte offset to start reading from (default: 0)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("FILE TYPE PREDICTION")
    print("=" * 80)
    print()
    
    # Check if file exists
    if not os.path.exists(args.file):
        print(f"❌ Error: File not found: {args.file}")
        return
    
    file_size = os.path.getsize(args.file)
    print(f"📄 File: {args.file}")
    print(f"📊 Size: {file_size:,} bytes")
    print()
    
    # Load model and encoder
    print("Loading model...")
    model, label_encoder = load_model_and_encoder()
    if model is None or label_encoder is None:
        return
    print()
    
    # Predict
    print("Analyzing file...")
    result = predict_file_type(args.file, model, label_encoder)
    
    if result is None:
        return
    
    if len(result) == 3:
        predicted_type, confidence, top_predictions = result
        print()
        print("=" * 80)
        print("PREDICTION RESULTS")
        print("=" * 80)
        print()
        print(f"🎯 Predicted File Type: {predicted_type.upper()}")
        if confidence is not None:
            print(f"📊 Confidence: {confidence:.2f}%")
        print()
        
        if top_predictions:
            print("Top 3 Predictions:")
            print("-" * 40)
            for i, (file_type, prob) in enumerate(top_predictions, 1):
                bar_length = int(prob / 2)
                bar = "█" * bar_length
                print(f"{i}. {file_type:12s} {prob:6.2f}% {bar}")
    else:
        predicted_type = result
        print()
        print("=" * 80)
        print(f"🎯 Predicted File Type: {predicted_type.upper()}")
        print("=" * 80)
    
    print()

if __name__ == '__main__':
    # If no arguments provided, show usage
    if len(sys.argv) == 1:
        print("=" * 80)
        print("FILE TYPE PREDICTION TOOL")
        print("=" * 80)
        print()
        print("Usage:")
        print("  python predict_file_type.py <file_path>")
        print()
        print("Example:")
        print("  python predict_file_type.py myfile.bin")
        print("  python predict_file_type.py document.pdf")
        print("  python predict_file_type.py unknown_file --offset 1024")
        print()
        print("Supported file types:")
        print("  7zip, apk, bin, css, elf, html, javascript, json, mp3, mp4, pdf, tif")
        print()
        sys.exit(0)
    
    main()
