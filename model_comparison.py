#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Comparison Script

This script compares the performance of four age prediction models:
1. Simple Model
2. K-Fold Ensemble Model
3. Enhanced Simple Model
4. Enhanced K-Fold Ensemble Model

It creates visualizations for different comparison pairs and a comprehensive
comparison of all models together.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

# Create directory for comparison results
COMPARISON_DIR = "model_comparisons"
os.makedirs(COMPARISON_DIR, exist_ok=True)

# File paths for metrics
SIMPLE_MODEL_PATH = "results/simple_voyageapi_embedding_model_accuracy.txt"
KFOLD_MODEL_PATH = "results/k_fold_ensemble_accuracy.txt"
ENHANCED_SIMPLE_MODEL_PATH = "enhanced_models_results/enhanced_simple_voyageapi_accuracy.txt"
ENHANCED_KFOLD_MODEL_PATH = "enhanced_models_results/enhanced_k_fold_ensemble_accuracy.txt"

def extract_metrics_from_file(file_path):
    """Extract accuracy metrics from a file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
            # Extract model name from the first line if it exists
            first_line = content.strip().split('\n')[0]
            if "Accuracy Metrics" in first_line:
                model_name = "Unnamed Model"
            else:
                model_name = first_line.strip().rstrip(':')
            
            # Extract MAE
            mae_match = re.search(r'Mean Absolute Error: (\d+\.\d+)', content)
            mae = float(mae_match.group(1)) if mae_match else None
            
            # Extract accuracy within thresholds
            within_1_match = re.search(r'Predictions within ±1 year: (\d+\.\d+)', content)
            within_1 = float(within_1_match.group(1)) if within_1_match else None
            
            within_3_match = re.search(r'Predictions within ±3 years: (\d+\.\d+)', content)
            within_3 = float(within_3_match.group(1)) if within_3_match else None
            
            within_5_match = re.search(r'Predictions within ±5 years: (\d+\.\d+)', content)
            within_5 = float(within_5_match.group(1)) if within_5_match else None
            
            # Extract R² score
            r2_match = re.search(r'R² score: (\d+\.\d+)', content)
            r2 = float(r2_match.group(1)) if r2_match else None
            
            return {
                'name': model_name,
                'mae': mae,
                'within_1_year': within_1,
                'within_3_years': within_3,
                'within_5_years': within_5,
                'r2_score': r2
            }
    except Exception as e:
        print(f"Error extracting metrics from {file_path}: {e}")
        return None

def calculate_improvement(baseline_value, new_value, higher_is_better=True):
    """Calculate percentage improvement between two values."""
    if baseline_value == 0:
        return float('inf') if new_value > 0 else 0
    
    if higher_is_better:
        improvement = (new_value - baseline_value) / baseline_value * 100
    else:
        improvement = (baseline_value - new_value) / baseline_value * 100
        
    return improvement

def create_comparison_plot(model1, model2, metric_type, output_path=None):
    """Create a comparison bar plot for two models on a specific metric."""
    models = [model1['name'], model2['name']]
    
    if metric_type == 'mae':
        values = [model1['mae'], model2['mae']]
        title = 'Mean Absolute Error Comparison'
        ylabel = 'MAE (years)'
        higher_is_better = False
    elif metric_type == 'accuracy':
        # For accuracy, we'll plot all threshold values
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        
        # Within 1 year
        ax[0].bar([0, 1], [model1['within_1_year'], model2['within_1_year']])
        ax[0].set_xticks([0, 1])
        ax[0].set_xticklabels(models)
        ax[0].set_ylabel('Accuracy (%)')
        ax[0].set_title('Predictions within ±1 year')
        
        # Within 3 years
        ax[1].bar([0, 1], [model1['within_3_years'], model2['within_3_years']])
        ax[1].set_xticks([0, 1])
        ax[1].set_xticklabels(models)
        ax[1].set_ylabel('Accuracy (%)')
        ax[1].set_title('Predictions within ±3 years')
        
        # Within 5 years
        ax[2].bar([0, 1], [model1['within_5_years'], model2['within_5_years']])
        ax[2].set_xticks([0, 1])
        ax[2].set_xticklabels(models)
        ax[2].set_ylabel('Accuracy (%)')
        ax[2].set_title('Predictions within ±5 years')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
            
        return
        
    elif metric_type == 'r2':
        values = [model1['r2_score'], model2['r2_score']]
        title = 'R² Score Comparison'
        ylabel = 'R² Score'
        higher_is_better = True
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar([0, 1], values)
    
    # Add value annotations
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{values[i]:.4f}', ha='center', va='bottom')
    
    # Determine which model performs better and highlight it
    if values[0] == values[1]:
        # Values are equal, no highlighting
        pass
    elif (values[0] > values[1]) == higher_is_better:
        # Model 1 is better
        bars[0].set_color('green')
        bars[1].set_color('lightgray')
        improvement = calculate_improvement(values[1], values[0], higher_is_better)
        plt.text(0, values[0] * 1.1, f"+{improvement:.2f}%" if improvement > 0 else f"{improvement:.2f}%", 
                 ha='center', va='bottom', color='green', fontweight='bold')
    else:
        # Model 2 is better
        bars[0].set_color('lightgray')
        bars[1].set_color('green')
        improvement = calculate_improvement(values[0], values[1], higher_is_better)
        plt.text(1, values[1] * 1.1, f"+{improvement:.2f}%" if improvement > 0 else f"{improvement:.2f}%", 
                 ha='center', va='bottom', color='green', fontweight='bold')
    
    plt.xticks([0, 1], models)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    if output_path:
        plt.savefig(output_path)
        plt.close()

def create_comprehensive_comparison(models, output_path=None):
    """Create a comprehensive comparison of all models."""
    model_names = [model['name'] for model in models]
    x = np.arange(len(model_names))
    width = 0.15
    
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 2, figure=fig)
    
    # MAE subplot
    ax1 = fig.add_subplot(gs[0, :])
    mae_values = [model['mae'] for model in models]
    bars1 = ax1.bar(x, mae_values, width, label='MAE')
    ax1.set_ylabel('MAE (years)')
    ax1.set_title('Mean Absolute Error (lower is better)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names)
    
    # Add value annotations for MAE
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')
    
    # R² subplot
    ax2 = fig.add_subplot(gs[1, :])
    r2_values = [model['r2_score'] for model in models]
    bars2 = ax2.bar(x, r2_values, width, label='R²', color='orange')
    ax2.set_ylabel('R² Score')
    ax2.set_title('R² Score (higher is better)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names)
    
    # Add value annotations for R²
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.annotate(f'{height:.4f}',
                     xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')
    
    # Accuracy within thresholds
    ax3 = fig.add_subplot(gs[2, 0])
    width = 0.25
    within_1_values = [model['within_1_year'] for model in models]
    within_3_values = [model['within_3_years'] for model in models]
    within_5_values = [model['within_5_years'] for model in models]
    
    bars3_1 = ax3.bar(x - width, within_1_values, width, label='±1 year')
    bars3_3 = ax3.bar(x, within_3_values, width, label='±3 years')
    bars3_5 = ax3.bar(x + width, within_5_values, width, label='±5 years')
    
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Prediction Accuracy within Thresholds')
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_names)
    ax3.legend()
    
    # Add value annotations for accuracy thresholds
    for bars, values in [(bars3_1, within_1_values), 
                         (bars3_3, within_3_values), 
                         (bars3_5, within_5_values)]:
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax3.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    # Summary table
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis('tight')
    ax4.axis('off')
    
    # Create a dataframe for the table
    summary_data = {
        'Model': model_names,
        'MAE': mae_values,
        'R²': r2_values,
        '±1 year': [f"{v:.1f}%" for v in within_1_values],
        '±3 years': [f"{v:.1f}%" for v in within_3_values],
        '±5 years': [f"{v:.1f}%" for v in within_5_values]
    }
    summary_df = pd.DataFrame(summary_data)
    
    # Create table
    table = ax4.table(cellText=summary_df.values, colLabels=summary_df.columns, 
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    ax4.set_title('Model Performance Summary')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()

def create_pairwise_comparison_plots(models_dict):
    """Create pairwise comparison plots between models."""
    # Define the pairs to compare
    comparison_pairs = [
        ('simple', 'kfold', 'Simple vs K-Fold'),
        ('simple', 'enhanced_simple', 'Simple vs Enhanced Simple'),
        ('enhanced_simple', 'enhanced_kfold', 'Enhanced Simple vs Enhanced K-Fold'),
        ('kfold', 'enhanced_kfold', 'K-Fold vs Enhanced K-Fold')
    ]
    
    for model1_key, model2_key, title in comparison_pairs:
        model1 = models_dict[model1_key]
        model2 = models_dict[model2_key]
        
        # Update model names for better display
        model1['name'] = model1_key.replace('_', ' ').title() + ' Model'
        model2['name'] = model2_key.replace('_', ' ').title() + ' Model'
        
        # Create comparison directory
        pair_dir = os.path.join(COMPARISON_DIR, f"{model1_key}_vs_{model2_key}")
        os.makedirs(pair_dir, exist_ok=True)
        
        # Create MAE comparison
        create_comparison_plot(
            model1, model2, 'mae',
            os.path.join(pair_dir, f"mae_comparison.png")
        )
        
        # Create R² comparison
        create_comparison_plot(
            model1, model2, 'r2',
            os.path.join(pair_dir, f"r2_comparison.png")
        )
        
        # Create accuracy comparison
        create_comparison_plot(
            model1, model2, 'accuracy',
            os.path.join(pair_dir, f"accuracy_comparison.png")
        )

def create_html_report(models_dict):
    """Create an HTML report with all comparisons."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Age Prediction Models Comparison</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
            }
            h1, h2, h3 {
                color: #2c3e50;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }
            th, td {
                padding: 12px 15px;
                border: 1px solid #ddd;
                text-align: center;
            }
            th {
                background-color: #f2f2f2;
                font-weight: bold;
            }
            tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            .section {
                margin-bottom: 30px;
                padding: 20px;
                border-radius: 5px;
                background-color: #fff;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .comparison-images {
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                gap: 20px;
                margin-top: 20px;
            }
            .comparison-images img {
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 4px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .highlight-better {
                color: green;
                font-weight: bold;
            }
            .highlight-worse {
                color: #d35400;
            }
            .full-width-image {
                width: 100%;
                max-width: 1000px;
                margin: 20px auto;
                display: block;
            }
        </style>
    </head>
    <body>
        <h1>Age Prediction Models Comparison</h1>
        
        <div class="section">
            <h2>Summary of All Models</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>MAE (years) ↓</th>
                    <th>R² Score ↑</th>
                    <th>±1 year (%) ↑</th>
                    <th>±3 years (%) ↑</th>
                    <th>±5 years (%) ↑</th>
                </tr>
    """
    
    # Add model data to table
    for key, model in models_dict.items():
        model_name = key.replace('_', ' ').title() + ' Model'
        html_content += f"""
                <tr>
                    <td>{model_name}</td>
                    <td>{model['mae']:.2f}</td>
                    <td>{model['r2_score']:.4f}</td>
                    <td>{model['within_1_year']:.1f}%</td>
                    <td>{model['within_3_years']:.1f}%</td>
                    <td>{model['within_5_years']:.1f}%</td>
                </tr>
        """
    
    html_content += """
            </table>
            
            <h3>Comprehensive Comparison</h3>
            <img src="all_models_comparison.png" alt="All Models Comparison" class="full-width-image">
        </div>
    """
    
    # Add pairwise comparisons
    comparison_pairs = [
        ('simple', 'kfold', 'Simple vs K-Fold Model'),
        ('simple', 'enhanced_simple', 'Simple vs Enhanced Simple Model'),
        ('enhanced_simple', 'enhanced_kfold', 'Enhanced Simple vs Enhanced K-Fold Model'),
        ('kfold', 'enhanced_kfold', 'K-Fold vs Enhanced K-Fold Model')
    ]
    
    for model1_key, model2_key, title in comparison_pairs:
        pair_dir_rel = f"{model1_key}_vs_{model2_key}"
        
        html_content += f"""
        <div class="section">
            <h2>{title}</h2>
            
            <h3>Performance Metrics Comparison</h3>
            <div class="comparison-images">
                <img src="{pair_dir_rel}/mae_comparison.png" alt="MAE Comparison">
                <img src="{pair_dir_rel}/r2_comparison.png" alt="R² Comparison">
            </div>
            
            <h3>Accuracy Within Thresholds</h3>
            <img src="{pair_dir_rel}/accuracy_comparison.png" alt="Accuracy Comparison" class="full-width-image">
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    # Write HTML report
    with open(os.path.join(COMPARISON_DIR, "model_comparison_report.html"), 'w') as f:
        f.write(html_content)

def create_markdown_report(models_dict):
    """Create a markdown report with model comparisons."""
    md_content = "# Age Prediction Models Comparison\n\n"
    
    md_content += "## Summary of All Models\n\n"
    
    # Create markdown table
    md_content += "| Model | MAE (years) ↓ | R² Score ↑ | ±1 year (%) ↑ | ±3 years (%) ↑ | ±5 years (%) ↑ |\n"
    md_content += "| ----- | ------------ | ---------- | ------------- | -------------- | -------------- |\n"
    
    # Add model data to table
    for key, model in models_dict.items():
        model_name = key.replace('_', ' ').title() + ' Model'
        md_content += f"| {model_name} | {model['mae']:.2f} | {model['r2_score']:.4f} | {model['within_1_year']:.1f}% | {model['within_3_years']:.1f}% | {model['within_5_years']:.1f}% |\n"
    
    md_content += "\n## Model Comparisons\n\n"
    
    # Add pairwise comparisons
    comparison_pairs = [
        ('simple', 'kfold', 'Simple vs K-Fold Model'),
        ('simple', 'enhanced_simple', 'Simple vs Enhanced Simple Model'),
        ('enhanced_simple', 'enhanced_kfold', 'Enhanced Simple vs Enhanced K-Fold Model'),
        ('kfold', 'enhanced_kfold', 'K-Fold vs Enhanced K-Fold Model')
    ]
    
    for model1_key, model2_key, title in comparison_pairs:
        model1 = models_dict[model1_key]
        model2 = models_dict[model2_key]
        model1_name = model1_key.replace('_', ' ').title() + ' Model'
        model2_name = model2_key.replace('_', ' ').title() + ' Model'
        
        md_content += f"### {title}\n\n"
        
        # Compare MAE
        mae_diff = calculate_improvement(model1['mae'], model2['mae'], higher_is_better=False)
        if model1['mae'] < model2['mae']:
            mae_comparison = f"**{model1_name}** has a lower MAE ({model1['mae']:.2f} vs {model2['mae']:.2f}), making it **{abs(mae_diff):.2f}%** better."
        elif model2['mae'] < model1['mae']:
            mae_comparison = f"**{model2_name}** has a lower MAE ({model2['mae']:.2f} vs {model1['mae']:.2f}), making it **{abs(mae_diff):.2f}%** better."
        else:
            mae_comparison = f"Both models have the same MAE ({model1['mae']:.2f})."
        
        md_content += f"- **Mean Absolute Error**: {mae_comparison}\n"
        
        # Compare R²
        r2_diff = calculate_improvement(model1['r2_score'], model2['r2_score'], higher_is_better=True)
        if model1['r2_score'] > model2['r2_score']:
            r2_comparison = f"**{model1_name}** has a higher R² score ({model1['r2_score']:.4f} vs {model2['r2_score']:.4f}), making it **{abs(r2_diff):.2f}%** better."
        elif model2['r2_score'] > model1['r2_score']:
            r2_comparison = f"**{model2_name}** has a higher R² score ({model2['r2_score']:.4f} vs {model1['r2_score']:.4f}), making it **{abs(r2_diff):.2f}%** better."
        else:
            r2_comparison = f"Both models have the same R² score ({model1['r2_score']:.4f})."
        
        md_content += f"- **R² Score**: {r2_comparison}\n"
        
        # Compare accuracy thresholds
        for threshold, key in [('±1 year', 'within_1_year'), ('±3 years', 'within_3_years'), ('±5 years', 'within_5_years')]:
            threshold_diff = calculate_improvement(model1[key], model2[key], higher_is_better=True)
            if model1[key] > model2[key]:
                threshold_comparison = f"**{model1_name}** has better accuracy {threshold} ({model1[key]:.1f}% vs {model2[key]:.1f}%), making it **{abs(threshold_diff):.2f}%** better."
            elif model2[key] > model1[key]:
                threshold_comparison = f"**{model2_name}** has better accuracy {threshold} ({model2[key]:.1f}% vs {model1[key]:.1f}%), making it **{abs(threshold_diff):.2f}%** better."
            else:
                threshold_comparison = f"Both models have the same accuracy {threshold} ({model1[key]:.1f}%)."
            
            md_content += f"- **Accuracy {threshold}**: {threshold_comparison}\n"
        
        md_content += "\n"
    
    md_content += "## Conclusion\n\n"
    
    # Find best model for each metric
    best_mae = min(models_dict.items(), key=lambda x: x[1]['mae'])
    best_r2 = max(models_dict.items(), key=lambda x: x[1]['r2_score'])
    best_within_1 = max(models_dict.items(), key=lambda x: x[1]['within_1_year'])
    best_within_3 = max(models_dict.items(), key=lambda x: x[1]['within_3_years'])
    best_within_5 = max(models_dict.items(), key=lambda x: x[1]['within_5_years'])
    
    md_content += "Based on the comparison, here are the best models for each metric:\n\n"
    md_content += f"- **Best MAE**: {best_mae[0].replace('_', ' ').title()} Model ({best_mae[1]['mae']:.2f} years)\n"
    md_content += f"- **Best R² Score**: {best_r2[0].replace('_', ' ').title()} Model ({best_r2[1]['r2_score']:.4f})\n"
    md_content += f"- **Best ±1 year Accuracy**: {best_within_1[0].replace('_', ' ').title()} Model ({best_within_1[1]['within_1_year']:.1f}%)\n"
    md_content += f"- **Best ±3 years Accuracy**: {best_within_3[0].replace('_', ' ').title()} Model ({best_within_3[1]['within_3_years']:.1f}%)\n"
    md_content += f"- **Best ±5 years Accuracy**: {best_within_5[0].replace('_', ' ').title()} Model ({best_within_5[1]['within_5_years']:.1f}%)\n\n"
    
    # Write markdown report
    with open(os.path.join(COMPARISON_DIR, "model_comparison_report.md"), 'w') as f:
        f.write(md_content)

def main():
    """Main function to run the model comparison."""
    print("Extracting metrics from model files...")
    
    # Extract metrics from files
    simple_model = extract_metrics_from_file(SIMPLE_MODEL_PATH)
    kfold_model = extract_metrics_from_file(KFOLD_MODEL_PATH)
    enhanced_simple_model = extract_metrics_from_file(ENHANCED_SIMPLE_MODEL_PATH)
    enhanced_kfold_model = extract_metrics_from_file(ENHANCED_KFOLD_MODEL_PATH)
    
    # Organize models in a dictionary
    models_dict = {
        'simple': simple_model,
        'kfold': kfold_model,
        'enhanced_simple': enhanced_simple_model,
        'enhanced_kfold': enhanced_kfold_model
    }
    
    # Create pairwise comparison plots
    print("Creating pairwise comparison plots...")
    create_pairwise_comparison_plots(models_dict)
    
    # Create comprehensive comparison
    print("Creating comprehensive comparison...")
    all_models = [simple_model, kfold_model, enhanced_simple_model, enhanced_kfold_model]
    for i, model_key in enumerate(['simple', 'kfold', 'enhanced_simple', 'enhanced_kfold']):
        all_models[i]['name'] = model_key.replace('_', ' ').title() + ' Model'
    
    create_comprehensive_comparison(
        all_models, 
        os.path.join(COMPARISON_DIR, "all_models_comparison.png")
    )
    
    # Create HTML report
    print("Creating HTML report...")
    create_html_report(models_dict)
    
    # Create markdown report
    print("Creating markdown report...")
    create_markdown_report(models_dict)
    
    print(f"Model comparison complete. Results saved to '{COMPARISON_DIR}' directory.")

if __name__ == "__main__":
    main()
