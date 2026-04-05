"""
Batch file type prediction script supporting multiple models.
Predicts file types for all files in a directory using trained models.
"""

import os
import sys
import pickle
import numpy as np
import json
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import tensorflow as tf

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')


class ModelPredictor:
    """Unified predictor for all model types."""
    
    def __init__(self, model_name: str):
        """
        Initialize predictor with specified model.
        
        Args:
            model_name: Name of model ('rf', 'xgboost', 'cnn', 'resnet')
        """
        self.model_name = model_name
        self.model = None
        self.label_encoder = None
        self.model_display_name = self._get_display_name()
        
    def _get_display_name(self) -> str:
        """Get display name for model."""
        names = {
            'rf': 'Random Forest',
            'xgboost': 'XGBoost',
            'cnn': 'CNN',
            'resnet': 'ResNet'
        }
        return names.get(self.model_name, self.model_name.upper())
    
    def load_model(self) -> bool:
        """
        Load model and label encoder.
        
        Returns:
            True if successful, False otherwise
        """
        # Load label encoder
        encoder_path = Path('results/label_encoder.pkl')
        if not encoder_path.exists():
            print(f"❌ Error: Label encoder not found at {encoder_path}")
            return False
        
        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Load model based on type
        if self.model_name in ['rf', 'xgboost']:
            return self._load_sklearn_model()
        elif self.model_name in ['cnn', 'resnet']:
            return self._load_keras_model()
        else:
            print(f"❌ Error: Unknown model type: {self.model_name}")
            return False
    
    def _load_sklearn_model(self) -> bool:
        """Load scikit-learn based model (Random Forest or XGBoost)."""
        model_paths = {
            'rf': Path('results/random_forest/rf_model.pkl'),
            'xgboost': Path('results/xgboost/xgb_model.pkl')
        }
        
        model_path = model_paths.get(self.model_name)
        if not model_path or not model_path.exists():
            print(f"❌ Error: Model not found at {model_path}")
            print(f"   Please train the model first: python models/{self.model_name}/train.py")
            return False
        
        with open(model_path, 'rb') as f:
            model_wrapper = pickle.load(f)
        
        # Extract the actual sklearn model
        self.model = model_wrapper.model if hasattr(model_wrapper, 'model') else model_wrapper
        return True
    
    def _load_keras_model(self) -> bool:
        """Load Keras based model (CNN or ResNet)."""
        model_paths = {
            'cnn': Path('results/cnn/cnn_model.h5'),
            'resnet': Path('results/resnet/resnet_model.h5')
        }
        
        model_path = model_paths.get(self.model_name)
        if not model_path or not model_path.exists():
            print(f"❌ Error: Model not found at {model_path}")
            print(f"   Please train the model first: python models/{self.model_name}/train.py")
            return False
        
        self.model = tf.keras.models.load_model(model_path)
        return True
    
    def predict(self, file_path: str, fragment_size: int = 4096) -> Tuple[str, float]:
        """
        Predict file type.
        
        Args:
            file_path: Path to file
            fragment_size: Size of fragment to read
            
        Returns:
            Tuple of (predicted_type, confidence)
        """
        # Read file fragment
        fragment = self._read_fragment(file_path, fragment_size)
        if fragment is None:
            return "error", 0.0
        
        # Predict based on model type
        if self.model_name in ['rf', 'xgboost']:
            return self._predict_sklearn(fragment)
        else:
            return self._predict_keras(fragment)
    
    def _read_fragment(self, file_path: str, fragment_size: int) -> np.ndarray:
        """Read file fragment."""
        try:
            with open(file_path, 'rb') as f:
                data = f.read(fragment_size)
            
            if len(data) < fragment_size:
                data = data + b'\x00' * (fragment_size - len(data))
            
            return np.frombuffer(data, dtype=np.uint8)
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
            return None
    
    def _predict_sklearn(self, fragment: np.ndarray) -> Tuple[str, float]:
        """Predict using sklearn model."""
        fragment = fragment.reshape(1, -1)
        
        # Get prediction
        prediction = self.model.predict(fragment)
        predicted_label = self.label_encoder.inverse_transform(prediction)[0]
        
        # Get confidence
        try:
            proba = self.model.predict_proba(fragment)[0]
            confidence = np.max(proba) * 100
        except:
            confidence = 0.0
        
        return predicted_label, confidence
    
    def _predict_keras(self, fragment: np.ndarray) -> Tuple[str, float]:
        """Predict using Keras model."""
        fragment = fragment.reshape(1, -1, 1)
        
        # Get prediction
        proba = self.model.predict(fragment, verbose=0)[0]
        predicted_idx = np.argmax(proba)
        predicted_label = self.label_encoder.inverse_transform([predicted_idx])[0]
        confidence = proba[predicted_idx] * 100
        
        return predicted_label, confidence


def load_files_from_directory(directory: str) -> List[str]:
    """
    Load all files from directory.
    
    Args:
        directory: Path to directory
        
    Returns:
        List of file paths
    """
    directory = Path(directory)
    if not directory.exists():
        print(f"❌ Error: Directory not found: {directory}")
        return []
    
    files = []
    for file_path in directory.iterdir():
        if file_path.is_file():
            files.append(str(file_path))
    
    return sorted(files)


def print_predictions_table(predictions: List[Dict], model_name: str):
    """
    Print predictions in formatted table.
    
    Args:
        predictions: List of prediction dictionaries
        model_name: Display name of model
    """
    print("\n" + "="*70)
    print(f"🤖 {model_name} Predictions")
    print("="*70)
    
    # Print header
    print(f"{'File':<40} {'Prediction':<12} {'Confidence':<10}")
    print("-"*70)
    
    # Print predictions
    for pred in predictions:
        file_name = Path(pred['file']).name
        prediction = pred['prediction']
        confidence = pred['confidence']
        print(f"{file_name:<40} {prediction:<12} {confidence:>5.1f}%")
    
    print()


def save_predictions_report(predictions: List[Dict], model_name: str, output_dir: str = 'results/predictions'):
    """
    Save predictions to JSON and CSV.
    
    Args:
        predictions: List of prediction dictionaries
        model_name: Name of model
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    json_path = output_path / f'{model_name}_predictions.json'
    with open(json_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    # Save CSV
    csv_path = output_path / f'{model_name}_predictions.csv'
    with open(csv_path, 'w') as f:
        f.write("file,prediction,confidence\n")
        for pred in predictions:
            f.write(f"{pred['file']},{pred['prediction']},{pred['confidence']:.2f}\n")
    
    print(f"📄 Predictions saved to {json_path} and {csv_path}")


def generate_comparison_report(all_predictions: Dict[str, List[Dict]], output_dir: str = 'results/predictions'):
    """
    Generate comparison report across all models.
    
    Args:
        all_predictions: Dictionary mapping model names to predictions
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all files
    files = set()
    for predictions in all_predictions.values():
        for pred in predictions:
            files.add(Path(pred['file']).name)
    
    files = sorted(files)
    
    # Create comparison table
    report_path = output_path / 'comparison_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*120 + "\n")
        f.write("FILE TYPE PREDICTION COMPARISON REPORT\n")
        f.write("="*120 + "\n\n")
        
        # Header
        f.write(f"{'File':<30}")
        for model_name in all_predictions.keys():
            f.write(f"{model_name:<25}")
        f.write("\n")
        f.write("-"*120 + "\n")
        
        # Predictions
        for file in files:
            f.write(f"{file:<30}")
            for model_name, predictions in all_predictions.items():
                pred = next((p for p in predictions if Path(p['file']).name == file), None)
                if pred:
                    f.write(f"{pred['prediction']:<12} ({pred['confidence']:>5.1f}%) ")
                else:
                    f.write(f"{'N/A':<25}")
            f.write("\n")
        
        f.write("\n")
    
    print(f"📄 Comparison report saved to {report_path}")


def generate_file_type_report(all_predictions: Dict[str, List[Dict]], output_dir: str = 'results/predictions'):
    """
    Generate per-file-type accuracy report.
    
    Args:
        all_predictions: Dictionary mapping model names to predictions
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Group predictions by expected file type
    file_type_results = {}
    
    for model_name, predictions in all_predictions.items():
        for pred in predictions:
            file_name = Path(pred['file']).name
            
            # Extract expected type from filename (e.g., "7zip.bin" -> "7zip")
            expected_type = file_name.rsplit('.', 1)[0]
            
            # Map common variations
            type_mapping = {
                '7zip': '.7zip',
                'apk': '.apk',
                'bin': '.bin',
                'css': '.css',
                'dll': '.dll',
                'elf': '.elf',
                'html': '.html',
                'js': '.javascript',
                'json': '.json',
                'mp3': '.mp3',
                'mp4': '.mp4',
                'pdf': '.pdf',
                'rtf': '.rtf',
                'swf': '.swf',
                'tar': '.tar',
                'tif': '.tif',
                'xls': '.xlsx',
                'xlsx': '.xlsx'
            }
            
            expected = type_mapping.get(expected_type, f'.{expected_type}')
            
            if expected not in file_type_results:
                file_type_results[expected] = {}
            
            if model_name not in file_type_results[expected]:
                file_type_results[expected][model_name] = []
            
            is_correct = pred['prediction'] == expected
            file_type_results[expected][model_name].append({
                'file': file_name,
                'predicted': pred['prediction'],
                'confidence': pred['confidence'],
                'correct': is_correct
            })
    
    # Generate report
    report_path = output_path / 'file_type_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("PER FILE TYPE PREDICTION REPORT\n")
        f.write("="*100 + "\n\n")
        
        for file_type in sorted(file_type_results.keys()):
            f.write(f"\n{'='*100}\n")
            f.write(f"FILE TYPE: {file_type.upper()}\n")
            f.write(f"{'='*100}\n\n")
            
            for model_name in sorted(file_type_results[file_type].keys()):
                results = file_type_results[file_type][model_name]
                correct = sum(1 for r in results if r['correct'])
                total = len(results)
                accuracy = (correct / total * 100) if total > 0 else 0
                
                f.write(f"{model_name}:\n")
                f.write(f"  Accuracy: {accuracy:.1f}% ({correct}/{total} correct)\n")
                f.write(f"  Predictions:\n")
                
                for result in results:
                    status = "✓" if result['correct'] else "✗"
                    f.write(f"    {status} {result['file']:<30} -> {result['predicted']:<12} ({result['confidence']:>5.1f}%)\n")
                
                f.write("\n")
    
    print(f"📄 File type report saved to {report_path}")


def main():
    """Main prediction pipeline."""
    parser = argparse.ArgumentParser(
        description='Batch file type prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py predict_input/ --model rf
  python predict.py predict_input/ --model cnn
  python predict.py predict_input/ --model all
  python predict.py predict_input/ --model all --save-report
        """
    )
    
    parser.add_argument('directory', type=str, help='Directory containing files to predict')
    parser.add_argument('--model', type=str, default='all',
                       choices=['rf', 'xgboost', 'cnn', 'resnet', 'all'],
                       help='Model to use for prediction (default: all)')
    parser.add_argument('--save-report', action='store_true',
                       help='Save prediction reports')
    parser.add_argument('--comparison', action='store_true',
                       help='Generate comparison report (requires --model all)')
    
    args = parser.parse_args()
    
    # Load files
    files = load_files_from_directory(args.directory)
    if not files:
        return
    
    print(f"📁 Loaded {len(files)} fragment(s) from: {args.directory}\n")
    
    # Determine which models to run
    if args.model == 'all':
        models = ['rf', 'xgboost', 'cnn', 'resnet']
    else:
        models = [args.model]
    
    all_predictions = {}
    
    # Run predictions for each model
    for model_name in models:
        predictor = ModelPredictor(model_name)
        
        print(f"Loading {predictor.model_display_name} model...")
        if not predictor.load_model():
            continue
        
        # Make predictions
        predictions = []
        for file_path in files:
            pred_type, confidence = predictor.predict(file_path)
            predictions.append({
                'file': file_path,
                'prediction': pred_type,
                'confidence': confidence
            })
        
        # Print and save results
        print_predictions_table(predictions, predictor.model_display_name)
        
        if args.save_report:
            save_predictions_report(predictions, model_name)
        
        all_predictions[predictor.model_display_name] = predictions
    
    # Generate comparison reports if requested
    if args.comparison and len(all_predictions) > 1:
        print("\nGenerating comparison reports...")
        generate_comparison_report(all_predictions)
        generate_file_type_report(all_predictions)


if __name__ == '__main__':
    main()
