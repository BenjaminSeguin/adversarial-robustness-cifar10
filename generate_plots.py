#!/usr/bin/env python3
"""
Comprehensive Evaluation and Visualization Script
=================================================
Generates publication-quality plots for adversarial defense analysis.

This script:
1. Evaluates all models across different attack strengths (epsilon)
2. Tests individual models, pruned models, and ensemble
3. Generates all recommended plots from literature
4. Saves results and figures

Usage:
    python generate_plots.py --all                    # Generate all plots
    python generate_plots.py --quick                  # Quick evaluation (fewer epsilons)
    python generate_plots.py --plot-only              # Only plot existing results
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import torch
import torchvision
import torchvision.transforms as transforms

# Import from model.py
from model import Net, EnsembleNet, fgsm_attack, pgd_attack, randomization_layer

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

valid_size = 1024
batch_size = 32


def get_validation_loader(dataset, valid_size=1024, batch_size=32):
    """Get validation data loader"""
    indices = list(range(len(dataset)))
    valid_sampler = torch.utils.data.SubsetRandomSampler(indices[:valid_size])
    valid_loader = torch.utils.data.DataLoader(dataset, sampler=valid_sampler, batch_size=batch_size)
    return valid_loader


def evaluate_model(net, valid_loader, epsilon_values, use_randomization=False, num_samples=10):
    """
    Evaluate model across different epsilon values.
    
    Returns:
        dict with keys: 'natural', 'fgsm', 'pgd' 
        each containing list of accuracies for each epsilon
    """
    results = {
        'natural': [],
        'fgsm': [],
        'pgd': []
    }
    
    net.eval()
    
    for epsilon in epsilon_values:
        print(f"  Testing with epsilon={epsilon:.3f}")
        
        # Natural accuracy
        correct_nat = 0
        total_nat = 0
        
        # FGSM accuracy
        correct_fgsm = 0
        total_fgsm = 0
        
        # PGD accuracy
        correct_pgd = 0
        total_pgd = 0
        
        with torch.no_grad():
            for data in valid_loader:
                images, labels = data[0].to(device), data[1].to(device)
                
                # Test natural
                if use_randomization:
                    for _ in range(num_samples):
                        rand_images = randomization_layer(images)
                        outputs = net(rand_images)
                        _, predicted = torch.max(outputs.data, 1)
                        total_nat += labels.size(0)
                        correct_nat += (predicted == labels).sum().item()
                else:
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total_nat += labels.size(0)
                    correct_nat += (predicted == labels).sum().item()
        
        # FGSM attack
        for data in valid_loader:
            images, labels = data[0].to(device), data[1].to(device)
            adv_images = fgsm_attack(net, images, labels, epsilon)
            
            if use_randomization:
                for _ in range(num_samples):
                    rand_adv = randomization_layer(adv_images)
                    with torch.no_grad():
                        outputs = net(rand_adv)
                        _, predicted = torch.max(outputs.data, 1)
                        total_fgsm += labels.size(0)
                        correct_fgsm += (predicted == labels).sum().item()
            else:
                with torch.no_grad():
                    outputs = net(adv_images)
                    _, predicted = torch.max(outputs.data, 1)
                    total_fgsm += labels.size(0)
                    correct_fgsm += (predicted == labels).sum().item()
        
        # PGD attack
        for data in valid_loader:
            images, labels = data[0].to(device), data[1].to(device)
            adv_images = pgd_attack(net, images, labels, epsilon, alpha=0.01, num_iter=7)
            
            if use_randomization:
                for _ in range(num_samples):
                    rand_adv = randomization_layer(adv_images)
                    with torch.no_grad():
                        outputs = net(rand_adv)
                        _, predicted = torch.max(outputs.data, 1)
                        total_pgd += labels.size(0)
                        correct_pgd += (predicted == labels).sum().item()
            else:
                with torch.no_grad():
                    outputs = net(adv_images)
                    _, predicted = torch.max(outputs.data, 1)
                    total_pgd += labels.size(0)
                    correct_pgd += (predicted == labels).sum().item()
        
        # Compute accuracies
        if use_randomization:
            acc_nat = 100 * correct_nat / (total_nat)
            acc_fgsm = 100 * correct_fgsm / (total_fgsm)
            acc_pgd = 100 * correct_pgd / (total_pgd)
        else:
            acc_nat = 100 * correct_nat / total_nat
            acc_fgsm = 100 * correct_fgsm / total_fgsm
            acc_pgd = 100 * correct_pgd / total_pgd
        
        results['natural'].append(acc_nat)
        results['fgsm'].append(acc_fgsm)
        results['pgd'].append(acc_pgd)
        
        print(f"    Natural: {acc_nat:.2f}%, FGSM: {acc_fgsm:.2f}%, PGD: {acc_pgd:.2f}%")
    
    return results


def load_one_pixel_results():
    """Load one-pixel attack results from results_one_pixel/"""
    one_pixel_file = 'results_one_pixel_nov29/plots/results_one_pixel.json'
    
    if not os.path.exists(one_pixel_file):
        print(f"⊗ One-pixel results not found: {one_pixel_file}")
        return None
    
    try:
        with open(one_pixel_file, 'r') as f:
            data = json.load(f)
        
        # Extract one-pixel results at num_pixels=1 (first value in arrays)
        one_pixel_results = {}
        for model_name, results in data['results'].items():
            one_pixel_results[model_name] = results['one_pixel'][0]  # Take first value (1 pixel)
        
        print(f"✓ Loaded one-pixel attack results from {one_pixel_file}")
        return one_pixel_results
    except Exception as e:
        print(f"⊗ Error loading one-pixel results: {e}")
        return None


def run_evaluation(epsilon_values, quick=False):
    """Run complete evaluation of all models"""
    
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*70)
    
    # Load dataset
    cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=transforms.ToTensor())
    valid_loader = get_validation_loader(cifar, valid_size, batch_size)
    
    all_results = {}
    
    # Models to evaluate
    models_config = [
        ('default_model.pth', 'Default (No Defense)', False, False),
        ('fgsm_model.pth', 'FGSM Trained', False, False),
        ('pgd_model.pth', 'PGD Trained', False, False),
        ('mixed_model.pth', 'Mixed Trained', False, False),
        ('fgsm_model_pruned.pth', 'FGSM Pruned', False, False),
        ('pgd_model_pruned.pth', 'PGD Pruned', False, False),
        ('mixed_model_pruned.pth', 'Mixed Pruned', False, False),
        ('pgd_model_pruned.pth', 'PGD Pruned + Rand', True, False),
    ]
    
    # Evaluate individual models
    for model_file, name, use_rand, is_ensemble in models_config:
        full_path = os.path.join('models', model_file)
        
        if not os.path.exists(full_path):
            print(f"\n⊗ Skipping {name} - file not found: {full_path}")
            continue
        
        print(f"\n{'='*70}")
        print(f"Evaluating: {name}")
        print(f"{'='*70}")
        
        try:
            net = Net()
            net.load(full_path)
            net.to(device)
            net.eval()
            
            results = evaluate_model(net, valid_loader, epsilon_values, use_rand, num_samples=5 if quick else 10)
            all_results[name] = results
            
        except Exception as e:
            print(f"Error loading {name}: {e}")
    
    # Evaluate ensemble
    ensemble_models = [
        'models/fgsm_model.pth',
        'models/pgd_model.pth',
        'models/mixed_model.pth'
    ]
    
    if all(os.path.exists(m) for m in ensemble_models):
        print(f"\n{'='*70}")
        print("Evaluating: Ensemble (Adversarially Trained Models)")
        print(f"{'='*70}")
        
        try:
            ensemble = EnsembleNet(model_files=ensemble_models)
            ensemble.to(device)
            ensemble.eval()
            
            results = evaluate_model(ensemble, valid_loader, epsilon_values, False, num_samples=1)
            all_results['Ensemble'] = results
            
            print(f"\n{'='*70}")
            print("Evaluating: Ensemble + Randomization")
            print(f"{'='*70}")
            
            results_rand = evaluate_model(ensemble, valid_loader, epsilon_values, True, num_samples=5 if quick else 10)
            all_results['Ensemble + Rand'] = results_rand
            
        except Exception as e:
            print(f"Error with ensemble: {e}")
    else:
        print("\n⊗ Skipping ensemble - pruned models not found")
    
    # Load and merge one-pixel attack results
    one_pixel_results = load_one_pixel_results()
    if one_pixel_results:
        for model_name in all_results:
            if model_name in one_pixel_results:
                # Add one-pixel accuracy as a list (same length as epsilon_values)
                all_results[model_name]['one_pixel'] = [one_pixel_results[model_name]] * len(epsilon_values)
            else:
                # If no one-pixel data for this model, fill with None
                all_results[model_name]['one_pixel'] = [None] * len(epsilon_values)
    
    return all_results, epsilon_values


def plot_robustness_curves(all_results, epsilon_values, output_dir='plots'):
    """Plot 1 & 3: Robustness curves and Attack success rate"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot for each attack type
    attack_types = ['fgsm', 'pgd']
    attack_names = ['FGSM Attack', 'PGD Attack']
    
    for attack_type, attack_name in zip(attack_types, attack_names):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Accuracy vs Epsilon
        for model_name, results in all_results.items():
            if attack_type in results:
                ax1.plot(epsilon_values, results[attack_type], marker='o', label=model_name, linewidth=2)
        
        ax1.set_xlabel('Attack Strength (ε)', fontsize=12)
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.set_title(f'Robustness Curve: {attack_name}', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 100])
        
        # Plot 2: Attack Success Rate
        for model_name, results in all_results.items():
            if attack_type in results:
                success_rate = [100 - acc for acc in results[attack_type]]
                ax2.plot(epsilon_values, success_rate, marker='o', label=model_name, linewidth=2)
        
        ax2.set_xlabel('Attack Strength (ε)', fontsize=12)
        ax2.set_ylabel('Attack Success Rate (%)', fontsize=12)
        ax2.set_title(f'Attack Success Rate: {attack_name}', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 100])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'robustness_curve_{attack_type}.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_dir}/robustness_curve_{attack_type}.png")
        plt.close()


def plot_accuracy_comparison(all_results, epsilon_values, output_dir='plots'):
    """Plot 2: Bar chart comparing accuracies at fixed epsilon"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Use middle epsilon value for comparison
    eps_idx = len(epsilon_values) // 2
    epsilon = epsilon_values[eps_idx]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    models = list(all_results.keys())
    natural_acc = [all_results[m]['natural'][eps_idx] for m in models]
    fgsm_acc = [all_results[m]['fgsm'][eps_idx] for m in models]
    pgd_acc = [all_results[m]['pgd'][eps_idx] for m in models]
    
    # Add one-pixel if available
    has_one_pixel = any('one_pixel' in all_results[m] and all_results[m]['one_pixel'][0] is not None for m in models)
    if has_one_pixel:
        one_pixel_acc = [all_results[m].get('one_pixel', [None])[eps_idx] if all_results[m].get('one_pixel', [None])[eps_idx] is not None else 0 for m in models]
    
    x = np.arange(len(models))
    width = 0.2 if has_one_pixel else 0.25
    
    ax.bar(x - 1.5*width, natural_acc, width, label='Natural', color='#2ecc71')
    ax.bar(x - 0.5*width, fgsm_acc, width, label=f'FGSM (ε={epsilon})', color='#e74c3c')
    ax.bar(x + 0.5*width, pgd_acc, width, label=f'PGD (ε={epsilon})', color='#e67e22')
    
    if has_one_pixel:
        ax.bar(x + 1.5*width, one_pixel_acc, width, label='One-Pixel Attack', color='#9b59b6')
    
    ax.set_xlabel('Model / Defense', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Accuracy Comparison Across Defenses (ε={epsilon})', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison_bar.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/accuracy_comparison_bar.png")
    plt.close()


def plot_ablation_study(all_results, epsilon_values, output_dir='plots'):
    """Plot 5: Defense layer ablation study"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Define progression of defenses
    ablation_order = [
        'Default (No Defense)',
        'PGD Trained',
        'PGD Pruned',
        'Ensemble',
        'Ensemble + Rand'
    ]
    
    # Filter to only models in ablation order
    ablation_results = {k: v for k, v in all_results.items() if k in ablation_order}
    
    if len(ablation_results) < 2:
        print("⊗ Not enough models for ablation study")
        return
    
    # Use middle epsilon
    eps_idx = len(epsilon_values) // 2
    epsilon = epsilon_values[eps_idx]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # FGSM ablation
    models = [m for m in ablation_order if m in ablation_results]
    fgsm_acc = [ablation_results[m]['fgsm'][eps_idx] for m in models]
    
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(models)))
    ax1.barh(models, fgsm_acc, color=colors)
    ax1.set_xlabel('Accuracy (%)', fontsize=12)
    ax1.set_title(f'Defense Ablation Study: FGSM (ε={epsilon})', fontsize=14, fontweight='bold')
    ax1.set_xlim([0, 100])
    ax1.grid(True, alpha=0.3, axis='x')
    
    for i, v in enumerate(fgsm_acc):
        ax1.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=10)
    
    # PGD ablation
    pgd_acc = [ablation_results[m]['pgd'][eps_idx] for m in models]
    
    ax2.barh(models, pgd_acc, color=colors)
    ax2.set_xlabel('Accuracy (%)', fontsize=12)
    ax2.set_title(f'Defense Ablation Study: PGD (ε={epsilon})', fontsize=14, fontweight='bold')
    ax2.set_xlim([0, 100])
    ax2.grid(True, alpha=0.3, axis='x')
    
    for i, v in enumerate(pgd_acc):
        ax2.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ablation_study.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/ablation_study.png")
    plt.close()


def plot_efficiency_comparison(output_dir='plots'):
    """Plot 7: Model efficiency (size, speed, accuracy)"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get model file sizes
    models_info = []
    
    model_files = {
        'Default': 'models/default_model.pth',
        'PGD': 'models/pgd_model.pth',
        'PGD Pruned': 'models/pgd_model_pruned.pth',
    }
    
    for name, path in model_files.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            models_info.append((name, size_mb))
    
    if not models_info:
        print("⊗ No models found for efficiency comparison")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = [m[0] for m in models_info]
    sizes = [m[1] for m in models_info]
    
    colors = ['#3498db', '#e74c3c', '#2ecc71'][:len(names)]
    bars = ax.bar(names, sizes, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Model Size (MB)', fontsize=12)
    ax.set_title('Model Size Comparison (Pruning Effect)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, size in zip(bars, sizes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{size:.2f} MB',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_efficiency.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/model_efficiency.png")
    plt.close()


def save_results_table(all_results, epsilon_values, output_dir='plots'):
    """Save results as CSV and markdown table"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Use middle epsilon
    eps_idx = len(epsilon_values) // 2
    epsilon = epsilon_values[eps_idx]
    
    # Check if one-pixel data exists
    has_one_pixel = any('one_pixel' in all_results[m] and all_results[m]['one_pixel'][0] is not None for m in all_results)
    
    # Create markdown table
    md_content = f"# Evaluation Results Summary\n\n"
    md_content += f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md_content += f"## Accuracy at epsilon = {epsilon}\n\n"
    
    if has_one_pixel:
        md_content += "| Model | Natural (%) | FGSM (%) | PGD (%) | One-Pixel (%) |\n"
        md_content += "|-------|-------------|----------|---------|---------------|\n"
    else:
        md_content += "| Model | Natural (%) | FGSM (%) | PGD (%) |\n"
        md_content += "|-------|-------------|----------|--------|\n"
    
    for model_name, results in all_results.items():
        nat = results['natural'][eps_idx]
        fgsm = results['fgsm'][eps_idx]
        pgd = results['pgd'][eps_idx]
        
        if has_one_pixel and 'one_pixel' in results and results['one_pixel'][0] is not None:
            one_pix = results['one_pixel'][eps_idx]
            md_content += f"| {model_name} | {nat:.2f} | {fgsm:.2f} | {pgd:.2f} | {one_pix:.2f} |\n"
        elif has_one_pixel:
            md_content += f"| {model_name} | {nat:.2f} | {fgsm:.2f} | {pgd:.2f} | N/A |\n"
        else:
            md_content += f"| {model_name} | {nat:.2f} | {fgsm:.2f} | {pgd:.2f} |\n"
    
    # Save markdown with UTF-8 encoding
    with open(os.path.join(output_dir, 'results_summary.md'), 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"✓ Saved: {output_dir}/results_summary.md")
    
    # Save JSON
    results_data = {
        'epsilon_values': epsilon_values,
        'results': all_results,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2)
    print(f"✓ Saved: {output_dir}/results.json")


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation plots")
    parser.add_argument('--all', action='store_true', help="Generate all plots (default)")
    parser.add_argument('--quick', action='store_true', help="Quick evaluation with fewer epsilon values")
    parser.add_argument('--plot-only', action='store_true', help="Only generate plots from existing results.json")
    parser.add_argument('--output-dir', default='plots', help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Default to --all if nothing specified
    if not any([args.all, args.quick, args.plot_only]):
        args.all = True
    
    # Define epsilon values for evaluation
    if args.quick:
        epsilon_values = [0.0, 0.01, 0.03, 0.05]
    else:
        epsilon_values = [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08]
    
    # Run evaluation or load results
    if args.plot_only:
        results_file = os.path.join(args.output_dir, 'results.json')
        if not os.path.exists(results_file):
            print(f"Error: {results_file} not found. Run evaluation first.")
            return
        
        with open(results_file, 'r') as f:
            data = json.load(f)
            all_results = data['results']
            epsilon_values = data['epsilon_values']
        print(f"✓ Loaded results from {results_file}")
    else:
        all_results, epsilon_values = run_evaluation(epsilon_values, quick=args.quick)
    
    # Generate all plots
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70 + "\n")
    
    plot_robustness_curves(all_results, epsilon_values, args.output_dir)
    plot_accuracy_comparison(all_results, epsilon_values, args.output_dir)
    plot_ablation_study(all_results, epsilon_values, args.output_dir)
    # plot_efficiency_comparison(args.output_dir)  # Disabled - not needed
    
    # Save results
    if not args.plot_only:
        save_results_table(all_results, epsilon_values, args.output_dir)
    
    print("\n" + "="*70)
    print("✓ ALL PLOTS GENERATED SUCCESSFULLY!")
    print("="*70)
    print(f"\nPlots saved to: {args.output_dir}/")
    print("\nGenerated files:")
    print("  - robustness_curve_fgsm.png")
    print("  - robustness_curve_pgd.png")
    print("  - accuracy_comparison_bar.png")
    print("  - ablation_study.png")
    print("  - results_summary.md")
    print("  - results.json")


if __name__ == "__main__":
    main()
