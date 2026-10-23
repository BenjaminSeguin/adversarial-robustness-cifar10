#!/usr/bin/env python3
"""
Unified Attack Comparison Plot
===============================
Compares all models against all attack types: FGSM, PGD, and One-Pixel.

Usage:
    python plot_all_attacks.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

def load_results():
    """Load FGSM/PGD results and 1-pixel results"""
    
    # Load FGSM/PGD results
    with open('plots/results.json', 'r') as f:
        fgsm_pgd_data = json.load(f)
    
    # Load 1-pixel results
    with open('results_one_pixel/results_one_pixel.json', 'r') as f:
        one_pixel_data = json.load(f)
    
    return fgsm_pgd_data, one_pixel_data


def extract_accuracy_at_epsilon(results, epsilon=0.03):
    """Extract accuracy at specific epsilon value"""
    epsilon_values = results['epsilon_values']
    idx = epsilon_values.index(epsilon)
    
    accuracies = {}
    for model_name, model_results in results['results'].items():
        accuracies[model_name] = {
            'natural': model_results['natural'][idx],
            'fgsm': model_results['fgsm'][idx],
            'pgd': model_results['pgd'][idx]
        }
    
    return accuracies


def extract_one_pixel_accuracy(results, num_pixels=1):
    """Extract 1-pixel attack accuracy"""
    num_pixels_values = results['num_pixels_values']
    idx = num_pixels_values.index(num_pixels)
    
    accuracies = {}
    for model_name, model_results in results['results'].items():
        accuracies[model_name] = {
            'natural': model_results['natural'][idx],
            'one_pixel': model_results['one_pixel'][idx]
        }
    
    return accuracies


def plot_unified_comparison(fgsm_pgd_acc, one_pixel_acc, output_dir='plots'):
    """
    Create unified bar chart comparing all models against all attacks.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Use ALL models from FGSM/PGD results
    models = list(fgsm_pgd_acc.keys())
    natural_acc = [fgsm_pgd_acc[m]['natural'] for m in models]
    fgsm_acc = [fgsm_pgd_acc[m]['fgsm'] for m in models]
    pgd_acc = [fgsm_pgd_acc[m]['pgd'] for m in models]
    
    # Match 1-pixel data for each model
    pixel_acc = []
    for model in models:
        matched = False
        
        # Try exact match first
        if model in one_pixel_acc:
            pixel_acc.append(one_pixel_acc[model]['one_pixel'])
            matched = True
        else:
            # Try fuzzy matching
            model_lower = model.lower().replace(' ', '').replace('(', '').replace(')', '')
            for pixel_model in one_pixel_acc.keys():
                pixel_lower = pixel_model.lower().replace(' ', '').replace('(', '').replace(')', '')
                if model_lower in pixel_lower or pixel_lower in model_lower:
                    pixel_acc.append(one_pixel_acc[pixel_model]['one_pixel'])
                    matched = True
                    break
        
        if not matched:
            pixel_acc.append(None)  # No data available
    
    # Create plot
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x = np.arange(len(models))
    width = 0.18
    
    # Plot bars (skip 1-pixel if None)
    bars1 = ax.bar(x - 1.5*width, natural_acc, width, label='Natural', color='#2ecc71', alpha=0.9)
    bars2 = ax.bar(x - 0.5*width, fgsm_acc, width, label='FGSM (ε=0.03)', color='#e74c3c', alpha=0.9)
    bars3 = ax.bar(x + 0.5*width, pgd_acc, width, label='PGD (ε=0.03)', color='#e67e22', alpha=0.9)
    
    # Only plot 1-pixel bars where data exists
    pixel_heights = [p if p is not None else 0 for p in pixel_acc]
    bars4 = ax.bar(x + 1.5*width, pixel_heights, width, label='1-Pixel Attack', color='#9b59b6', alpha=0.9)
    
    # Customize plot
    ax.set_xlabel('Model / Defense', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Robustness Across Attack Types (ε=0.03, 1 pixel)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 100)
    
    # Add value labels on bars
    def add_labels(bars, values):
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if val is not None and height > 0:  # Only label if data exists
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=7)
    
    add_labels(bars1, natural_acc)
    add_labels(bars2, fgsm_acc)
    add_labels(bars3, pgd_acc)
    add_labels(bars4, pixel_acc)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'all_attacks_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_attack_heatmap(fgsm_pgd_acc, one_pixel_acc, output_dir='plots'):
    """
    Create a heatmap showing accuracy for each model vs attack type.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    models = list(fgsm_pgd_acc.keys())
    attacks = ['Natural', 'FGSM', 'PGD', '1-Pixel']
    
    data = []
    for model in models:
        row = [
            fgsm_pgd_acc[model]['natural'],
            fgsm_pgd_acc[model]['fgsm'],
            fgsm_pgd_acc[model]['pgd'],
            0  # Placeholder for 1-pixel
        ]
        
        # Try to find matching 1-pixel data
        for pixel_model in one_pixel_acc.keys():
            if model.lower().replace(' ', '') in pixel_model.lower().replace(' ', ''):
                row[3] = one_pixel_acc[pixel_model]['one_pixel']
                break
        
        data.append(row)
    
    data = np.array(data)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    # Set ticks
    ax.set_xticks(np.arange(len(attacks)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(attacks, fontsize=11, fontweight='bold')
    ax.set_yticklabels(models, fontsize=10)
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(attacks)):
            if data[i, j] > 0:
                text = ax.text(j, i, f'{data[i, j]:.1f}%',
                             ha="center", va="center", color="black", fontsize=9,
                             fontweight='bold')
    
    ax.set_title('Model Robustness Heatmap: Accuracy (%) Across Attack Types',
                fontsize=13, fontweight='bold', pad=15)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy (%)', rotation=270, labelpad=20, fontsize=11)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'attack_heatmap.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def main():
    print("="*70)
    print("UNIFIED ATTACK COMPARISON PLOT GENERATOR")
    print("="*70)
    
    # Load results
    print("\nLoading results...")
    fgsm_pgd_data, one_pixel_data = load_results()
    print("✓ Loaded FGSM/PGD results")
    print("✓ Loaded 1-Pixel results")
    
    # Extract accuracies at epsilon=0.03 and 1-pixel
    print("\nExtracting accuracies...")
    fgsm_pgd_acc = extract_accuracy_at_epsilon(fgsm_pgd_data, epsilon=0.03)
    one_pixel_acc = extract_one_pixel_accuracy(one_pixel_data, num_pixels=1)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_unified_comparison(fgsm_pgd_acc, one_pixel_acc)
    plot_attack_heatmap(fgsm_pgd_acc, one_pixel_acc)
    
    print("\n" + "="*70)
    print("✓ ALL PLOTS GENERATED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated files:")
    print("  - plots/all_attacks_comparison.png  (Bar chart)")
    print("  - plots/attack_heatmap.png          (Heatmap)")


if __name__ == "__main__":
    main()
