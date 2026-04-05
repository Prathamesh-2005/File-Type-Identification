"""
Comprehensive report generator for file type predictions.
Generates detailed reports showing performance across all models and file types.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class ReportGenerator:
    """Generate comprehensive reports for file type predictions."""
    
    def __init__(self, predictions_dir: str = 'results/predictions'):
        """
        Initialize report generator.
        
        Args:
            predictions_dir: Directory containing prediction results
        """
        self.predictions_dir = Path(predictions_dir)
        self.models = ['rf', 'xgboost', 'cnn', 'resnet']
        self.model_display_names = {
            'rf': 'Random Forest',
            'xgboost': 'XGBoost',
            'cnn': 'CNN',
            'resnet': 'ResNet'
        }
        self.file_type_mapping = {
            '7zip': '.7zip',
            'apk': '.apk',
            'bin': '.bin',
            'bmp': '.bmp',
            'css': '.css',
            'dll': '.dll',
            'elf': '.elf',
            'html': '.html',
            'js': '.javascript',
            'javascript': '.javascript',
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
        
    def load_predictions(self) -> Dict[str, List[Dict]]:
        """
        Load all prediction results.
        
        Returns:
            Dictionary mapping model names to prediction lists
        """
        all_predictions = {}
        
        for model in self.models:
            json_path = self.predictions_dir / f'{model}_predictions.json'
            if json_path.exists():
                with open(json_path, 'r') as f:
                    all_predictions[model] = json.load(f)
                print(f"✓ Loaded {model} predictions")
            else:
                print(f"⚠ Warning: {model} predictions not found at {json_path}")
        
        return all_predictions
    
    def get_expected_type(self, filename: str) -> str:
        """
        Extract expected file type from filename.
        
        Args:
            filename: Name of file
            
        Returns:
            Expected file type
        """
        # Remove extension and get base name
        base_name = Path(filename).stem
        
        return self.file_type_mapping.get(base_name, f'.{base_name}')
    
    def calculate_accuracy(self, predictions: List[Dict]) -> Dict:
        """
        Calculate accuracy metrics for predictions.
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            Dictionary with accuracy metrics
        """
        total = len(predictions)
        correct = 0
        by_file_type = {}
        
        for pred in predictions:
            filename = Path(pred['file']).name
            expected = self.get_expected_type(filename)
            predicted = pred['prediction']
            is_correct = predicted == expected
            
            if is_correct:
                correct += 1
            
            if expected not in by_file_type:
                by_file_type[expected] = {'correct': 0, 'total': 0}
            
            by_file_type[expected]['total'] += 1
            if is_correct:
                by_file_type[expected]['correct'] += 1
        
        return {
            'overall_accuracy': (correct / total * 100) if total > 0 else 0,
            'correct': correct,
            'total': total,
            'by_file_type': by_file_type
        }
    
    def generate_summary_report(self, all_predictions: Dict[str, List[Dict]]):
        """
        Generate summary report showing accuracy for all models.
        
        Args:
            all_predictions: Dictionary mapping model names to predictions
        """
        output_path = self.predictions_dir / 'summary_report.txt'
        
        with open(output_path, 'w') as f:
            f.write("="*100 + "\n")
            f.write("FILE TYPE PREDICTION SUMMARY REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*100 + "\n\n")
            
            # Overall accuracy table
            f.write("OVERALL ACCURACY\n")
            f.write("-"*100 + "\n")
            f.write(f"{'Model':<20} {'Accuracy':<15} {'Files Correct':<20} {'Total Files':<15}\n")
            f.write("-"*100 + "\n")
            
            for model in self.models:
                if model in all_predictions:
                    metrics = self.calculate_accuracy(all_predictions[model])
                    display_name = self.model_display_names[model]
                    accuracy = metrics['overall_accuracy']
                    correct = metrics['correct']
                    total = metrics['total']
                    
                    f.write(f"{display_name:<20} {accuracy:>6.2f}%{'':<8} {correct}/{total:<18} {total:<15}\n")
            
            f.write("\n\n")
            
            # Per file type accuracy
            f.write("PER FILE TYPE ACCURACY\n")
            f.write("="*100 + "\n\n")
            
            # Get all file types
            all_file_types = set()
            for predictions in all_predictions.values():
                for pred in predictions:
                    filename = Path(pred['file']).name
                    file_type = self.get_expected_type(filename)
                    all_file_types.add(file_type)
            
            for file_type in sorted(all_file_types):
                f.write(f"\n{file_type.upper()}\n")
                f.write("-"*100 + "\n")
                f.write(f"{'Model':<20} {'Accuracy':<15} {'Correct/Total':<20} {'Avg Confidence':<20}\n")
                f.write("-"*100 + "\n")
                
                for model in self.models:
                    if model in all_predictions:
                        metrics = self.calculate_accuracy(all_predictions[model])
                        display_name = self.model_display_names[model]
                        
                        if file_type in metrics['by_file_type']:
                            stats = metrics['by_file_type'][file_type]
                            accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
                            
                            # Calculate average confidence for this file type
                            confidences = []
                            for pred in all_predictions[model]:
                                if self.get_expected_type(Path(pred['file']).name) == file_type:
                                    confidences.append(pred['confidence'])
                            avg_conf = np.mean(confidences) if confidences else 0
                            
                            f.write(f"{display_name:<20} {accuracy:>6.2f}%{'':<8} {stats['correct']}/{stats['total']:<18} {avg_conf:>6.2f}%{'':<14}\n")
                
                f.write("\n")
        
        print(f"📄 Summary report saved to {output_path}")
    
    def generate_detailed_report(self, all_predictions: Dict[str, List[Dict]]):
        """
        Generate detailed report with all predictions.
        
        Args:
            all_predictions: Dictionary mapping model names to predictions
        """
        output_path = self.predictions_dir / 'detailed_report.txt'
        
        # Get all files
        all_files = set()
        for predictions in all_predictions.values():
            for pred in predictions:
                all_files.add(Path(pred['file']).name)
        
        with open(output_path, 'w') as f:
            f.write("="*120 + "\n")
            f.write("DETAILED FILE-BY-FILE PREDICTIONS\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*120 + "\n\n")
            
            for filename in sorted(all_files):
                expected = self.get_expected_type(filename)
                
                f.write(f"\n{'='*120}\n")
                f.write(f"FILE: {filename}\n")
                f.write(f"Expected Type: {expected}\n")
                f.write(f"{'='*120}\n\n")
                
                f.write(f"{'Model':<20} {'Prediction':<15} {'Confidence':<12} {'Status':<10}\n")
                f.write("-"*120 + "\n")
                
                for model in self.models:
                    if model in all_predictions:
                        pred = next((p for p in all_predictions[model] 
                                   if Path(p['file']).name == filename), None)
                        
                        if pred:
                            display_name = self.model_display_names[model]
                            prediction = pred['prediction']
                            confidence = pred['confidence']
                            status = "✓ Correct" if prediction == expected else "✗ Wrong"
                            
                            f.write(f"{display_name:<20} {prediction:<15} {confidence:>6.2f}%{'':<5} {status:<10}\n")
                
                f.write("\n")
        
        print(f"📄 Detailed report saved to {output_path}")
    
    def generate_confusion_matrices(self, all_predictions: Dict[str, List[Dict]]):
        """
        Generate confusion matrices for each model.
        
        Args:
            all_predictions: Dictionary mapping model names to predictions
        """
        from sklearn.metrics import confusion_matrix
        
        for model in self.models:
            if model not in all_predictions:
                continue
            
            predictions = all_predictions[model]
            
            # Get expected and predicted labels
            y_true = []
            y_pred = []
            
            for pred in predictions:
                filename = Path(pred['file']).name
                expected = self.get_expected_type(filename)
                predicted = pred['prediction']
                
                y_true.append(expected)
                y_pred.append(predicted)
            
            # Get unique labels
            labels = sorted(set(y_true + y_pred))
            
            # Compute confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            
            # Plot
            plt.figure(figsize=(14, 12))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=labels, yticklabels=labels,
                       cbar_kws={'label': 'Count'})
            plt.title(f'{self.model_display_names[model]} - Confusion Matrix', 
                     fontsize=16, fontweight='bold')
            plt.xlabel('Predicted', fontsize=12, fontweight='bold')
            plt.ylabel('Actual', fontsize=12, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            output_path = self.predictions_dir / f'{model}_confusion_matrix.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Confusion matrix saved to {output_path}")
    
    def generate_comparison_chart(self, all_predictions: Dict[str, List[Dict]]):
        """
        Generate comparison chart showing accuracy across models.
        
        Args:
            all_predictions: Dictionary mapping model names to predictions
        """
        # Calculate metrics for each model
        model_names = []
        accuracies = []
        
        for model in self.models:
            if model in all_predictions:
                metrics = self.calculate_accuracy(all_predictions[model])
                model_names.append(self.model_display_names[model])
                accuracies.append(metrics['overall_accuracy'])
        
        # Create bar chart
        plt.figure(figsize=(12, 6))
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        bars = plt.bar(model_names, accuracies, color=colors[:len(model_names)])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.xlabel('Model', fontsize=12, fontweight='bold')
        plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        plt.ylim([0, 105])
        plt.grid(axis='y', alpha=0.3)
        
        output_path = self.predictions_dir / 'model_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Comparison chart saved to {output_path}")
    
    def generate_per_filetype_chart(self, all_predictions: Dict[str, List[Dict]]):
        """
        Generate per-file-type accuracy chart.
        
        Args:
            all_predictions: Dictionary mapping model names to predictions
        """
        # Get all file types
        all_file_types = set()
        for predictions in all_predictions.values():
            for pred in predictions:
                filename = Path(pred['file']).name
                file_type = self.get_expected_type(filename)
                all_file_types.add(file_type)
        
        file_types = sorted(all_file_types)
        
        # Calculate accuracy for each model and file type
        data = {model: [] for model in self.models if model in all_predictions}
        
        for file_type in file_types:
            for model in self.models:
                if model in all_predictions:
                    metrics = self.calculate_accuracy(all_predictions[model])
                    
                    if file_type in metrics['by_file_type']:
                        stats = metrics['by_file_type'][file_type]
                        accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
                        data[model].append(accuracy)
                    else:
                        data[model].append(0)
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(16, 8))
        
        x = np.arange(len(file_types))
        width = 0.2
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        for i, (model, accuracies) in enumerate(data.items()):
            offset = width * (i - len(data)/2 + 0.5)
            ax.bar(x + offset, accuracies, width, 
                  label=self.model_display_names[model],
                  color=colors[i % len(colors)])
        
        ax.set_xlabel('File Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Per File Type Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(file_types, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 105])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.predictions_dir / 'per_filetype_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Per-file-type chart saved to {output_path}")
    
    def generate_all_reports(self):
        """Generate all reports."""
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE REPORTS")
        print("="*80 + "\n")
        
        # Load predictions
        print("Loading predictions...")
        all_predictions = self.load_predictions()
        
        if not all_predictions:
            print("❌ Error: No predictions found. Run predict.py first.")
            return
        
        print(f"\n✓ Loaded predictions for {len(all_predictions)} model(s)\n")
        
        # Generate reports
        print("Generating summary report...")
        self.generate_summary_report(all_predictions)
        
        print("\nGenerating detailed report...")
        self.generate_detailed_report(all_predictions)
        
        print("\nGenerating comparison chart...")
        self.generate_comparison_chart(all_predictions)
        
        print("\nGenerating per-file-type chart...")
        self.generate_per_filetype_chart(all_predictions)
        
        print("\nGenerating confusion matrices...")
        self.generate_confusion_matrices(all_predictions)
        
        print("\n" + "="*80)
        print("✅ ALL REPORTS GENERATED SUCCESSFULLY")
        print("="*80)
        print(f"\nReports saved to: {self.predictions_dir.absolute()}")


def main():
    """Main report generation pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate comprehensive reports for file type predictions'
    )
    parser.add_argument('--predictions-dir', type=str, default='results/predictions',
                       help='Directory containing prediction results')
    
    args = parser.parse_args()
    
    generator = ReportGenerator(args.predictions_dir)
    generator.generate_all_reports()


if __name__ == '__main__':
    main()
