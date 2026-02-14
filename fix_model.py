"""
Quick fix script to re-save the Random Forest model properly.
"""
import pickle
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    # Import the class definition
    from models.random_forest.train import RandomForestFileClassifier
    
    # Load the old model (with class wrapper)
    print("Loading existing model...")
    with open('results/random_forest/rf_model.pkl', 'rb') as f:
        wrapper = pickle.load(f)
    
    # Extract just the sklearn model
    sklearn_model = wrapper.model
    print(f"Extracted sklearn model: {type(sklearn_model)}")
    
    # Re-save just the sklearn model
    print("Re-saving model properly...")
    with open('results/random_forest/rf_model.pkl', 'wb') as f:
        pickle.dump(sklearn_model, f)
    
    print("✅ Model fixed! You can now use the Streamlit app.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
