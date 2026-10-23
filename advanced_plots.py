#!/usr/bin/env python3
"""
Advanced Visualizations
=======================
Generate additional advanced plots for comprehensive analysis:
- Adversarial example visualization
- Confusion matrices
- Ensemble diversity analysis
- Transferability heatmap

Usage:
    python advanced_plots.py --all
    python advanced_plots.py --adversarial-examples
    python advanced_plots.py --confusion-matrix
    python advanced_plots.py --diversity
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torchvision
import torchvision.transforms as transforms

from model import Net, EnsembleNet, fgsm_attack, pgd_attack, randomization_layer

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']


def plot_adversarial_examples(output_dir='plots'):
    """Visualize original vs adversarial images"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("Generating Adversarial Example Visualizations")
    print("="*70)
    
    # Load data and model
    cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=transforms.ToTensor())
    
    # Get a few test images
    indices = [10, 20, 30, 40, 50, 60]  # Mix of different classes
    images_list = []
    labels_list = []
    
    for idx in indices:
        img, label = cifar[idx]
        images_list.append(img)
        labels_list.append(label)
    
    images = torch.stack(images_list).to(device)
    labels = torch.tensor(labels_list).to(device)
    
    # Load model
    net = Net()
    if os.path.exists('models/pgd_model.pth'):
        net.load('models/pgd_model.pth')
    else:
        print("Warning: Using default model")
        net.load('models/default_model.pth')
    net.to(device)
    net.eval()
    
    # Generate adversarial examples
    epsilon = 0.03
    fgsm_adv = fgsm_attack(net, images, labels, epsilon)
    pgd_adv = pgd_attack(net, images, labels, epsilon, alpha=0.01, num_iter=7)
    
    # Get predictions
    with torch.no_grad():
        orig_pred = torch.argmax(net(images), dim=1)
        fgsm_pred = torch.argmax(net(fgsm_adv), dim=1)
        pgd_pred = torch.argmax(net(pgd_adv), dim=1)
    
    # Plot
    fig, axes = plt.subplots(len(indices), 4, figsize=(12, 3*len(indices)))
    
    for i in range(len(indices)):
        # Original image
        img_orig = images[i].cpu().permute(1, 2, 0).numpy()
        img_orig = np.clip(img_orig, 0, 1)
        
        axes[i, 0].imshow(img_orig)
        axes[i, 0].set_title(f'Original\n{CIFAR10_CLASSES[labels[i]]}\nPred: {CIFAR10_CLASSES[orig_pred[i]]}', 
                            fontsize=9, color='green' if orig_pred[i]==labels[i] else 'red')
        axes[i, 0].axis('off')
        
        # FGSM adversarial
        img_fgsm = fgsm_adv[i].cpu().permute(1, 2, 0).numpy()
        img_fgsm = np.clip(img_fgsm, 0, 1)
        
        axes[i, 1].imshow(img_fgsm)
        axes[i, 1].set_title(f'FGSM (ε={epsilon})\nPred: {CIFAR10_CLASSES[fgsm_pred[i]]}', 
                            fontsize=9, color='green' if fgsm_pred[i]==labels[i] else 'red')
        axes[i, 1].axis('off')
        
        # PGD adversarial
        img_pgd = pgd_adv[i].cpu().permute(1, 2, 0).numpy()
        img_pgd = np.clip(img_pgd, 0, 1)
        
        axes[i, 2].imshow(img_pgd)
        axes[i, 2].set_title(f'PGD (ε={epsilon})\nPred: {CIFAR10_CLASSES[pgd_pred[i]]}', 
                            fontsize=9, color='green' if pgd_pred[i]==labels[i] else 'red')
        axes[i, 2].axis('off')
        
        # Perturbation visualization (amplified)
        perturbation = (fgsm_adv[i] - images[i]).cpu().permute(1, 2, 0).numpy()
        perturbation_vis = (perturbation - perturbation.min()) / (perturbation.max() - perturbation.min() + 1e-8)
        
        axes[i, 3].imshow(perturbation_vis)
        axes[i, 3].set_title('Perturbation\n(FGSM, amplified)', fontsize=9)
        axes[i, 3].axis('off')
    
    plt.suptitle('Adversarial Examples Visualization', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'adversarial_examples.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/adversarial_examples.png")
    plt.close()


def plot_confusion_matrix(model_path, model_name, attack_type='pgd', epsilon=0.03, output_dir='plots'):
    """Generate confusion matrix for a model under attack"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating confusion matrix for {model_name} under {attack_type.upper()} attack...")
    
    # Load data
    cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=transforms.ToTensor())
    valid_size = 1024
    indices = list(range(len(cifar)))[:valid_size]
    valid_sampler = torch.utils.data.SubsetRandomSampler(indices)
    valid_loader = torch.utils.data.DataLoader(cifar, sampler=valid_sampler, batch_size=32)
    
    # Load model
    net = Net()
    net.load(model_path)
    net.to(device)
    net.eval()
    
    # Collect predictions
    all_preds = []
    all_labels = []
    
    for images, labels in valid_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Generate adversarial examples
        if attack_type == 'fgsm':
            adv_images = fgsm_attack(net, images, labels, epsilon)
        elif attack_type == 'pgd':
            adv_images = pgd_attack(net, images, labels, epsilon, alpha=0.01, num_iter=7)
        else:
            adv_images = images
        
        with torch.no_grad():
            outputs = net(adv_images)
            _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Create confusion matrix
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CIFAR10_CLASSES, yticklabels=CIFAR10_CLASSES,
                ax=ax, cbar_kws={'label': 'Count'})
    
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_title(f'Confusion Matrix: {model_name}\n{attack_type.upper()} Attack (ε={epsilon})', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    filename = f'confusion_matrix_{model_name.replace(" ", "_").lower()}_{attack_type}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/{filename}")
    plt.close()


def plot_ensemble_diversity(output_dir='plots'):
    """Analyze and visualize ensemble diversity"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("Analyzing Ensemble Diversity")
    print("="*70)
    
    # Load ensemble models
    model_files = [
        'models/fgsm_model_pruned.pth',
        'models/pgd_model_pruned.pth',
        'models/mixed_model_pruned.pth'
    ]
    
    if not all(os.path.exists(f) for f in model_files):
        print("⊗ Ensemble models not found")
        return
    
    # Load models
    models = []
    model_names = ['FGSM', 'PGD', 'Mixed']
    
    for mfile in model_files:
        net = Net()
        net.load(mfile)
        net.to(device)
        net.eval()
        models.append(net)
    
    # Load test data
    cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=transforms.ToTensor())
    valid_size = 500
    indices = list(range(len(cifar)))[:valid_size]
    valid_sampler = torch.utils.data.SubsetRandomSampler(indices)
    valid_loader = torch.utils.data.DataLoader(cifar, sampler=valid_sampler, batch_size=32)
    
    # Generate adversarial examples and collect predictions
    all_predictions = [[] for _ in range(len(models))]
    
    epsilon = 0.03
    for images, labels in valid_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Generate PGD adversarial examples
        adv_images = pgd_attack(models[0], images, labels, epsilon, alpha=0.01, num_iter=7)
        
        # Get predictions from each model
        with torch.no_grad():
            for i, model in enumerate(models):
                outputs = model(adv_images)
                _, preds = torch.max(outputs, 1)
                all_predictions[i].extend(preds.cpu().numpy())
    
    # Compute agreement matrix
    n_models = len(models)
    agreement_matrix = np.zeros((n_models, n_models))
    
    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                agreement_matrix[i, j] = 100.0
            else:
                preds_i = np.array(all_predictions[i])
                preds_j = np.array(all_predictions[j])
                agreement = 100 * np.mean(preds_i == preds_j)
                agreement_matrix[i, j] = agreement
    
    # Plot agreement matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(agreement_matrix, annot=True, fmt='.1f', cmap='RdYlGn_r',
                xticklabels=model_names, yticklabels=model_names,
                ax=ax, vmin=0, vmax=100, cbar_kws={'label': 'Agreement (%)'})
    
    ax.set_title('Ensemble Model Diversity\n(Lower agreement = Higher diversity)', 
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ensemble_diversity.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/ensemble_diversity.png")
    plt.close()
    
    # Print diversity stats
    print("\nDiversity Statistics:")
    print(f"  Average agreement: {np.mean(agreement_matrix[np.triu_indices(n_models, k=1)]):.1f}%")
    print(f"  Min agreement: {np.min(agreement_matrix[np.triu_indices(n_models, k=1)]):.1f}%")
    print(f"  Max agreement: {np.max(agreement_matrix[np.triu_indices(n_models, k=1)]):.1f}%")


def plot_transferability_heatmap(output_dir='plots'):
    """Analyze attack transferability between models"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("Analyzing Attack Transferability")
    print("="*70)
    
    # Models to test
    model_configs = [
        ('models/default_model.pth', 'Default'),
        ('models/fgsm_model.pth', 'FGSM'),
        ('models/pgd_model.pth', 'PGD'),
    ]
    
    models = []
    model_names = []
    
    for path, name in model_configs:
        if os.path.exists(path):
            net = Net()
            net.load(path)
            net.to(device)
            net.eval()
            models.append(net)
            model_names.append(name)
    
    if len(models) < 2:
        print("⊗ Need at least 2 models for transferability analysis")
        return
    
    # Load test data
    cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=transforms.ToTensor())
    valid_size = 200
    indices = list(range(len(cifar)))[:valid_size]
    valid_sampler = torch.utils.data.SubsetRandomSampler(indices)
    valid_loader = torch.utils.data.DataLoader(cifar, sampler=valid_sampler, batch_size=32)
    
    # Compute transferability matrix
    n_models = len(models)
    transfer_matrix = np.zeros((n_models, n_models))
    epsilon = 0.03
    
    for i, source_model in enumerate(models):
        print(f"  Generating attacks from {model_names[i]}...")
        
        for j, target_model in enumerate(models):
            success_count = 0
            total_count = 0
            
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                
                # Generate attack on source model
                adv_images = pgd_attack(source_model, images, labels, epsilon, alpha=0.01, num_iter=7)
                
                # Test on target model
                with torch.no_grad():
                    outputs = target_model(adv_images)
                    _, preds = torch.max(outputs, 1)
                    
                    # Count misclassifications (successful attacks)
                    success_count += (preds != labels).sum().item()
                    total_count += labels.size(0)
            
            success_rate = 100 * success_count / total_count
            transfer_matrix[i, j] = success_rate
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(transfer_matrix, annot=True, fmt='.1f', cmap='Reds',
                xticklabels=model_names, yticklabels=model_names,
                ax=ax, vmin=0, vmax=100, cbar_kws={'label': 'Attack Success Rate (%)'})
    
    ax.set_ylabel('Attack Generated On ↓', fontsize=12)
    ax.set_xlabel('Attack Tested On →', fontsize=12)
    ax.set_title(f'Attack Transferability Heatmap\nPGD Attack (ε={epsilon})', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'transferability_heatmap.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/transferability_heatmap.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate advanced plots")
    parser.add_argument('--all', action='store_true', help="Generate all advanced plots")
    parser.add_argument('--adversarial-examples', action='store_true', help="Visualize adversarial examples")
    parser.add_argument('--confusion-matrix', action='store_true', help="Generate confusion matrices")
    parser.add_argument('--diversity', action='store_true', help="Analyze ensemble diversity")
    parser.add_argument('--transferability', action='store_true', help="Analyze attack transferability")
    parser.add_argument('--output-dir', default='plots', help="Output directory")
    
    args = parser.parse_args()
    
    # Default to all if nothing specified
    if not any([args.all, args.adversarial_examples, args.confusion_matrix, 
                args.diversity, args.transferability]):
        args.all = True
    
    print("\n" + "="*70)
    print("ADVANCED VISUALIZATION GENERATION")
    print("="*70)
    
    if args.all or args.adversarial_examples:
        plot_adversarial_examples(args.output_dir)
    
    if args.all or args.confusion_matrix:
        if os.path.exists('models/pgd_model.pth'):
            plot_confusion_matrix('models/pgd_model.pth', 'PGD Model', 'pgd', 0.03, args.output_dir)
        if os.path.exists('models/default_model.pth'):
            plot_confusion_matrix('models/default_model.pth', 'Default Model', 'pgd', 0.03, args.output_dir)
    
    if args.all or args.diversity:
        plot_ensemble_diversity(args.output_dir)
    
    # if args.all or args.transferability:  # Disabled - not needed
    #     plot_transferability_heatmap(args.output_dir)
    
    print("\n" + "="*70)
    print("✓ ADVANCED PLOTS COMPLETED!")
    print("="*70)
    print(f"\nPlots saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
