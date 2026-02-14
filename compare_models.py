"""
Model comparison and visualization script.
Generates comprehensive comparison charts across all models.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

def load_all_results():
    """Load results from all models."""
    results_dir = Path('results')
    results = {}
    
    model_names = ['random_forest', 'xgboost', 'cnn', 'resnet']
    display_names = ['Random Forest', 'XGBoost', 'CNN', 'ResNet']
    
    for model_name, display_name in zip(model_names, display_names):
        results_file = results_dir / model_name / 'metrics.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                results[display_name] = json.load(f)
    
    return results

def plot_overall_comparison(results, save_path):
    """Plot overall metrics comparison."""
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    data = {metric: [] for metric in metrics}
    
    for model in models:
        for metric in metrics:
            data[metric].append(results[model]['overall'][metric] * 100)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(models))
    width = 0.2
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        offset = width * (i - 1.5)
        bars = ax.bar(x + offset, data[metric], width, label=label, color=colors[i])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Overall Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='lower right')
    ax.set_ylim([0, 105])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 Overall comparison saved to {save_path}")

def plot_per_class_accuracy(results, save_path):
    """Plot per-class accuracy comparison."""
    models = list(results.keys())
    
    # Get all file types
    file_types = list(results[models[0]]['per_class'].keys())
    file_types.sort()
    
    # Collect data
    data = {model: [] for model in models}
    
    for model in models:
        for file_type in file_types:
            if file_type in results[model]['per_class']:
                acc = results[model]['per_class'][file_type]['accuracy'] * 100
                data[model].append(acc)
            else:
                data[model].append(0)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x = np.arange(len(file_types))
    width = 0.2
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for i, model in enumerate(models):
        offset = width * (i - 1.5)
        ax.bar(x + offset, data[model], width, label=model, color=colors[i])
    
    ax.set_xlabel('File Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(file_types, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 105])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 Per-class accuracy comparison saved to {save_path}")

def plot_per_class_heatmap(results, save_path):
    """Plot heatmap of per-class accuracy."""
    models = list(results.keys())
    
    # Get all file types
    file_types = list(results[models[0]]['per_class'].keys())
    file_types.sort()
    
    # Create matrix
    matrix = []
    for model in models:
        row = []
        for file_type in file_types:
            if file_type in results[model]['per_class']:
                acc = results[model]['per_class'][file_type]['accuracy'] * 100
                row.append(acc)
            else:
                row.append(0)
        matrix.append(row)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 6))
    
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    # Set ticks
    ax.set_xticks(np.arange(len(file_types)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(file_types, rotation=45, ha='right')
    ax.set_yticklabels(models)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy (%)', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(file_types)):
            text = ax.text(j, i, f'{matrix[i][j]:.1f}',
                         ha='center', va='center', color='black', fontsize=9)
    
    ax.set_title('Per-Class Accuracy Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 Per-class heatmap saved to {save_path}")

def plot_training_time_comparison(results, save_path):
    """Plot training time comparison."""
    models = []
    times = []
    
    for model, data in results.items():
        models.append(model)
        if 'training_time' in data:
            times.append(data['training_time'])
        else:
            times.append(0)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    bars = ax.bar(models, times, color=colors[:len(models)])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}s',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 Training time comparison saved to {save_path}")

def generate_comparison_report(results, save_path):
    """Generate text report of comparison."""
    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall metrics
        f.write("OVERALL METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Model':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
        f.write("-" * 80 + "\n")
        
        for model, data in results.items():
            metrics = data['overall']
            f.write(f"{model:<15} "
                   f"{metrics['accuracy']*100:>10.2f}%  "
                   f"{metrics['precision']*100:>10.2f}%  "
                   f"{metrics['recall']*100:>10.2f}%  "
                   f"{metrics['f1_score']*100:>10.2f}%\n")
        
        f.write("\n\n")
        
        # Per-class metrics
        file_types = list(results[list(results.keys())[0]]['per_class'].keys())
        file_types.sort()
        
        for file_type in file_types:
            f.write(f"\nFILE TYPE: {file_type.upper()}\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Model':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
            f.write("-" * 80 + "\n")
            
            for model in results.keys():
                if file_type in results[model]['per_class']:
                    metrics = results[model]['per_class'][file_type]
                    f.write(f"{model:<15} "
                           f"{metrics['accuracy']*100:>10.2f}%  "
                           f"{metrics['precision']*100:>10.2f}%  "
                           f"{metrics['recall']*100:>10.2f}%  "
                           f"{metrics['f1_score']*100:>10.2f}%\n")
            
            f.write("\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"📄 Comparison report saved to {save_path}")

def main():
    """Generate all comparison visualizations."""
    print("=" * 80)
    print("MODEL COMPARISON VISUALIZER")
    print("=" * 80)
    print()
    
    # Load results
    print("Loading results...")
    results = load_all_results()
    
    if not results:
        print("❌ No model results found. Train models first!")
        return
    
    print(f"✅ Loaded results for {len(results)} models: {', '.join(results.keys())}")
    print()
    
    # Create output directory
    output_dir = Path('results/comparisons')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    print("Generating comparison visualizations...")
    print()
    
    plot_overall_comparison(results, output_dir / 'overall_comparison.png')
    plot_per_class_accuracy(results, output_dir / 'per_class_accuracy.png')
    plot_per_class_heatmap(results, output_dir / 'per_class_heatmap.png')
    plot_training_time_comparison(results, output_dir / 'training_time.png')
    generate_comparison_report(results, output_dir / 'comparison_report.txt')
    
    print()
    print("=" * 80)
    print("✅ ALL COMPARISONS GENERATED SUCCESSFULLY")
    print(f"📁 Results saved to: {output_dir}")
    print("=" * 80)

if __name__ == '__main__':
    main()
