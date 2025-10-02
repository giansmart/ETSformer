"""
Análisis de resultados de ETSformer
Este script lee los logs JSON generados durante el entrenamiento y crea visualizaciones
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from glob import glob
import argparse

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_experiment_logs(checkpoint_path):
    """Carga todos los logs de experimentos de un directorio de checkpoints"""
    experiments = []
    
    for exp_dir in glob(os.path.join(checkpoint_path, "*")):
        if os.path.isdir(exp_dir):
            log_file = os.path.join(exp_dir, "training_log.json")
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    log_data = json.load(f)
                    log_data['experiment_dir'] = exp_dir
                    experiments.append(log_data)
    
    return experiments

def load_test_results(results_path):
    """Carga todos los resultados de test"""
    test_results = []
    
    for result_file in glob(os.path.join(results_path, "*/test_results_*.json")):
        with open(result_file, 'r') as f:
            test_data = json.load(f)
            test_data['file_path'] = result_file
            test_results.append(test_data)
    
    return test_results

def plot_training_curves(experiments, save_path='figures'):
    """Genera gráficas de curvas de entrenamiento"""
    os.makedirs(save_path, exist_ok=True)
    
    for exp in experiments:
        setting = exp['setting']
        epochs_data = pd.DataFrame(exp['epochs'])
        
        if len(epochs_data) == 0:
            continue
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Curves - {setting}', fontsize=16)
        
        # Loss curves
        ax = axes[0, 0]
        ax.plot(epochs_data['epoch'], epochs_data['train_loss'], label='Train', linewidth=2)
        ax.plot(epochs_data['epoch'], epochs_data['val_loss'], label='Validation', linewidth=2)
        ax.plot(epochs_data['epoch'], epochs_data['test_loss'], label='Test', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Learning rate
        ax = axes[0, 1]
        ax.plot(epochs_data['epoch'], epochs_data['learning_rate'], 'g-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Training time per epoch
        ax = axes[1, 0]
        ax.bar(epochs_data['epoch'], epochs_data['epoch_time'], color='skyblue')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Training Time per Epoch')
        ax.grid(True, alpha=0.3)
        
        # Loss comparison (last epoch)
        ax = axes[1, 1]
        last_epoch = epochs_data.iloc[-1]
        losses = [last_epoch['train_loss'], last_epoch['val_loss'], last_epoch['test_loss']]
        labels = ['Train', 'Validation', 'Test']
        colors = ['blue', 'orange', 'green']
        ax.bar(labels, losses, color=colors)
        ax.set_ylabel('Loss')
        ax.set_title('Final Loss Comparison')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'training_curves_{setting}.png'), dpi=300)
        plt.close()

def plot_metrics_comparison(test_results, save_path='figures'):
    """Compara métricas entre diferentes experimentos"""
    os.makedirs(save_path, exist_ok=True)
    
    if not test_results:
        print("No test results found")
        return
    
    # Preparar datos
    metrics_data = []
    for result in test_results:
        metrics = result['metrics']
        metrics['setting'] = result['setting'].split('_')[0:4]  # Simplificar nombre
        metrics['setting'] = '_'.join(metrics['setting'])
        metrics_data.append(metrics)
    
    df = pd.DataFrame(metrics_data)
    
    # Gráfica de comparación de métricas
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Metrics Comparison Across Experiments', fontsize=16)
    
    metrics_to_plot = ['mae', 'mse', 'rmse', 'mape', 'mspe']
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i // 3, i % 3]
        df_sorted = df.sort_values(by=metric)
        ax.bar(range(len(df_sorted)), df_sorted[metric])
        ax.set_xticks(range(len(df_sorted)))
        ax.set_xticklabels(df_sorted['setting'], rotation=45, ha='right')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} Comparison')
        ax.grid(True, alpha=0.3)
    
    # Eliminar el último subplot vacío
    fig.delaxes(axes[1, 2])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'metrics_comparison.png'), dpi=300)
    plt.close()

def generate_summary_report(experiments, test_results, save_path='figures'):
    """Genera un reporte resumen en formato Markdown"""
    os.makedirs(save_path, exist_ok=True)
    
    report_lines = []
    report_lines.append("# ETSformer Analysis Report")
    report_lines.append(f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Resumen de experimentos
    report_lines.append("## Experiments Summary\n")
    report_lines.append(f"Total experiments: {len(experiments)}\n")
    
    for exp in experiments:
        report_lines.append(f"\n### {exp['setting']}")
        config = exp['experiment_config']
        report_lines.append(f"- **Model**: {config['model']}")
        report_lines.append(f"- **Sequence Length**: {config['seq_len']}")
        report_lines.append(f"- **Prediction Length**: {config['pred_len']}")
        report_lines.append(f"- **Learning Rate**: {config['learning_rate']}")
        report_lines.append(f"- **Batch Size**: {config['batch_size']}")
        report_lines.append(f"- **Training Epochs**: {len(exp['epochs'])}/{config['train_epochs']}")
        report_lines.append(f"- **Early Stopped**: {exp.get('early_stopped', False)}")
        report_lines.append(f"- **Total Training Time**: {exp['total_training_time']:.2f} seconds")
        
        if exp['epochs']:
            best_val_loss = min([e['val_loss'] for e in exp['epochs']])
            report_lines.append(f"- **Best Validation Loss**: {best_val_loss:.6f}")
    
    # Resultados de test
    report_lines.append("\n## Test Results\n")
    
    if test_results:
        # Crear tabla de resultados
        report_lines.append("| Setting | MAE | MSE | RMSE | MAPE | MSPE |")
        report_lines.append("|---------|-----|-----|------|------|------|")
        
        for result in test_results:
            m = result['metrics']
            setting_short = '_'.join(result['setting'].split('_')[0:4])
            report_lines.append(
                f"| {setting_short} | {m['mae']:.4f} | {m['mse']:.4f} | "
                f"{m['rmse']:.4f} | {m['mape']:.4f} | {m['mspe']:.4f} |"
            )
    
    # Análisis de preprocesamiento
    report_lines.append("\n## Data Preprocessing and Loading\n")
    if experiments:
        exp = experiments[0]
        report_lines.append(f"- **Training samples**: {exp['train_samples']}")
        report_lines.append(f"- **Validation samples**: {exp['val_samples']}")
        report_lines.append(f"- **Test samples**: {exp['test_samples']}")
        report_lines.append(f"- **Features**: {exp['experiment_config']['features']}")
        report_lines.append(f"- **Target**: {exp['experiment_config']['target']}")
    
    # Implementación del modelo
    report_lines.append("\n## Model Implementation\n")
    if experiments:
        config = experiments[0]['experiment_config']
        report_lines.append(f"- **Model Architecture**: {config['model']}")
        report_lines.append(f"- **Encoder Layers**: {config['e_layers']}")
        report_lines.append(f"- **Decoder Layers**: {config['d_layers']}")
        report_lines.append(f"- **Model Dimension**: {config['d_model']}")
        report_lines.append(f"- **Number of Heads**: {config['n_heads']}")
        report_lines.append(f"- **Feedforward Dimension**: {config['d_ff']}")
        report_lines.append(f"- **Top-K Frequencies**: {config['K']}")
    
    # Guardar reporte
    report_path = os.path.join(save_path, 'analysis_report.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Report saved to {report_path}")

def plot_prediction_horizons_comparison(experiments, test_results, save_path='figures'):
    """Compara el rendimiento en diferentes horizontes de predicción"""
    os.makedirs(save_path, exist_ok=True)
    
    # Agrupar por horizonte de predicción
    horizon_metrics = {}
    
    for result in test_results:
        setting_parts = result['setting'].split('_')
        for part in setting_parts:
            if part.startswith('pl'):
                pred_len = int(part[2:])
                if pred_len not in horizon_metrics:
                    horizon_metrics[pred_len] = []
                horizon_metrics[pred_len].append(result['metrics'])
                break
    
    if not horizon_metrics:
        return
    
    # Preparar datos para graficar
    horizons = sorted(horizon_metrics.keys())
    metrics_names = ['mae', 'mse', 'rmse']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Performance vs Prediction Horizon', fontsize=16)
    
    for idx, metric in enumerate(metrics_names):
        ax = axes[idx]
        values = []
        for h in horizons:
            metric_values = [m[metric] for m in horizon_metrics[h]]
            values.append(np.mean(metric_values))
        
        ax.plot(horizons, values, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Prediction Horizon')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} vs Prediction Horizon')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'prediction_horizon_comparison.png'), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze ETSformer results')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', 
                       help='Path to checkpoints directory')
    parser.add_argument('--results', type=str, default='./results/', 
                       help='Path to results directory')
    parser.add_argument('--output', type=str, default='./analysis_output/', 
                       help='Path to save analysis outputs')
    args = parser.parse_args()
    
    print("Loading experiment logs...")
    experiments = load_experiment_logs(args.checkpoints)
    print(f"Found {len(experiments)} experiments")
    
    print("Loading test results...")
    test_results = load_test_results(args.results)
    print(f"Found {len(test_results)} test results")
    
    print("Generating training curves...")
    plot_training_curves(experiments, args.output)
    
    print("Generating metrics comparison...")
    plot_metrics_comparison(test_results, args.output)
    
    print("Generating prediction horizon comparison...")
    plot_prediction_horizons_comparison(experiments, test_results, args.output)
    
    print("Generating summary report...")
    generate_summary_report(experiments, test_results, args.output)
    
    print(f"\nAnalysis complete! Results saved to {args.output}")

if __name__ == "__main__":
    main()