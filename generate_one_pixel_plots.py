#!/usr/bin/env python3
"""
One Pixel Attack Evaluation and Visualization Script
====================================================
Generates publication-quality plots for One Pixel Attack analysis.

This script:
1. Evaluates models against One Pixel Attack with varying pixel counts
2. Uses warm-start optimization for efficiency:
   - Attacks in reverse order (5→4→3→2→1) to reuse pixel perturbations
   - Transfers attacks between similar models
3. Generates plots similar to FGSM/PGD evaluation
4. Saves results and figures efficiently

Usage:
    python generate_one_pixel_plots.py --all                    # Generate all plots
    python generate_one_pixel_plots.py --quick                  # Quick evaluation (fewer images)
    python generate_one_pixel_plots.py --plot-only              # Only plot existing results
    python generate_one_pixel_plots.py --max-images 200         # Limit to 200 images
    python generate_one_pixel_plots.py --models default pgd     # Only test specific models
    python generate_one_pixel_plots.py --no-warm-start          # Disable warm-start optimization
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm

import torch
import torchvision
import torchvision.transforms as transforms

# Import from model.py
from model import Net, EnsembleNet, one_pixel_attack

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

batch_size = 32


def get_validation_loader(dataset, valid_size=1024, batch_size=32):
    """Get validation data loader"""
    indices = list(range(len(dataset)))
    valid_sampler = torch.utils.data.SubsetRandomSampler(indices[:valid_size])
    valid_loader = torch.utils.data.DataLoader(dataset, sampler=valid_sampler, batch_size=batch_size)
    return valid_loader


def evaluate_model_one_pixel(net, valid_loader, num_pixels_values, max_images=None, 
                              pop_size=400, max_iter=100, use_warm_start=True,
                              previous_model_perturbations=None):
    """
    Evaluate model across different numbers of pixels for One Pixel Attack.
    
    OPTIMIZATION STRATEGY:
    - Attacks in REVERSE order (max_pixels → 1) to enable warm-starting
    - When going from N→(N-1) pixels, reuses the first (N-1) pixels from N-pixel attack
    - Can also warm-start from previous similar model
    
    Args:
        use_warm_start: If True, use perturbations from higher pixel counts and previous models
        previous_model_perturbations: Dict of {num_pixels: {global_img_idx: perturbation}} 
                                     from previous model evaluation
    
    Returns:
        results: dict with 'natural' and 'one_pixel' accuracies
        all_perturbations: dict with perturbations for each num_pixels (for warm-starting next model)
    """
    results = {
        'natural': [],
        'one_pixel': []
    }
    
    # We'll compute attacks in reverse order but store results in original order
    results_by_num_pixels = {}
    all_perturbations = {}  # Store perturbations for each num_pixels
    
    net.eval()
    
    # Natural accuracy (compute once)
    print("  Testing natural accuracy...")
    correct_nat = 0
    total_nat = 0
    
    images_processed = 0
    with torch.no_grad():
        for data in valid_loader:
            if max_images and images_processed >= max_images:
                break
            
            images, labels = data[0].to(device), data[1].to(device)
            
            # Limit batch size if needed
            if max_images:
                remaining = max_images - images_processed
                if remaining < images.size(0):
                    images = images[:remaining]
                    labels = labels[:remaining]
            
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total_nat += labels.size(0)
            correct_nat += (predicted == labels).sum().item()
            images_processed += images.size(0)
    
    acc_nat = 100 * correct_nat / total_nat
    print(f"    Natural accuracy: {acc_nat:.2f}%")
    
    # Attack in REVERSE order for warm-start efficiency: max → min
    reversed_pixel_values = sorted(num_pixels_values, reverse=True)
    
    print(f"  Attacking in optimized order: {' → '.join(map(str, reversed_pixel_values))}")
    
    previous_num_pixels_perturbations = None  # For warm-starting from N→(N-1) pixels
    
    for num_pixels in reversed_pixel_values:
        print(f"  Testing One Pixel Attack with num_pixels={num_pixels}")
        
        # Determine warm-start source
        init_perturbations = None
        warm_start_source = None
        
        if use_warm_start:
            # Priority 1: Warm-start from previous num_pixels (e.g., 5→4)
            if previous_num_pixels_perturbations is not None:
                init_perturbations = previous_num_pixels_perturbations
                warm_start_source = f"from {num_pixels+1}→{num_pixels} pixels"
            
            # Priority 2: Warm-start from previous model (same num_pixels)
            elif previous_model_perturbations is not None and num_pixels in previous_model_perturbations:
                init_perturbations = previous_model_perturbations[num_pixels]
                warm_start_source = f"from previous model"
        
        if warm_start_source:
            print(f"    Using warm-start {warm_start_source} ({len(init_perturbations)} perturbations)")
        
        correct_op = 0
        total_op = 0
        
        perturbations_for_this_num_pixels = {}
        
        images_processed = 0
        global_img_idx = 0  # Track absolute image index across batches
        
        for data in tqdm(valid_loader, desc=f"  One Pixel (n={num_pixels})", leave=False):
            if max_images and images_processed >= max_images:
                break
            
            images, labels = data[0].to(device), data[1].to(device)
            
            # Limit batch size if needed
            if max_images:
                remaining = max_images - images_processed
                if remaining < images.size(0):
                    images = images[:remaining]
                    labels = labels[:remaining]
            
            # Prepare initial perturbations for this batch
            batch_init_perts = None
            if init_perturbations is not None:
                batch_init_perts = {}
                for i in range(images.size(0)):
                    if global_img_idx + i in init_perturbations:
                        batch_init_perts[i] = init_perturbations[global_img_idx + i]
            
            # Generate adversarial examples with optional warm-start
            adv_images, batch_perturbations = one_pixel_attack(
                net, images, labels, num_pixels=num_pixels,
                pop_size=pop_size, max_iter=max_iter, verbose=False,
                initial_perturbations=batch_init_perts
            )
            
            # Store perturbations with global indices
            for local_idx, pert in batch_perturbations.items():
                perturbations_for_this_num_pixels[global_img_idx + local_idx] = pert
            
            # Test on adversarial examples
            with torch.no_grad():
                outputs = net(adv_images)
                _, predicted = torch.max(outputs.data, 1)
                total_op += labels.size(0)
                correct_op += (predicted == labels).sum().item()
            
            images_processed += images.size(0)
            global_img_idx += images.size(0)
        
        acc_op = 100 * correct_op / total_op
        results_by_num_pixels[num_pixels] = acc_op
        all_perturbations[num_pixels] = perturbations_for_this_num_pixels
        
        # Store for next iteration (N→N-1)
        previous_num_pixels_perturbations = perturbations_for_this_num_pixels
        
        print(f"    One Pixel Attack (n={num_pixels}): {acc_op:.2f}%")
    
    # Convert results back to original order
    for num_pixels in num_pixels_values:
        results['natural'].append(acc_nat)
        results['one_pixel'].append(results_by_num_pixels[num_pixels])
    
    return results, all_perturbations


def run_evaluation(num_pixels_values, quick=False, max_images=None, pop_size=400, 
                   max_iter=100, selected_models=None, use_warm_start=True):
    """Run complete evaluation of all models with warm-start optimization"""
    
    print("\n" + "="*70)
    print("ONE PIXEL ATTACK MODEL EVALUATION")
    if use_warm_start:
        print("Warm-start optimization: ENABLED")
        print(f"  - Attack order: {max(num_pixels_values)} → {min(num_pixels_values)} pixels")
        print(f"  - Transferring attacks between similar models")
    else:
        print("Warm-start optimization: DISABLED")
    print("="*70)
    
    if max_images:
        print(f"Maximum images per model: {max_images}")
    
    # Load dataset
    cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=transforms.ToTensor())
    valid_size = max_images if max_images else 1024
    valid_loader = get_validation_loader(cifar, valid_size, batch_size)
    
    all_results = {}
    
    # Models to evaluate - grouped by similarity for warm-start efficiency
    # Format: (file, name, short_name, similarity_group)
    models_config = [
        ('default_model.pth', 'Default (No Defense)', 'default', 'default'),
        ('fgsm_model.pth', 'FGSM Trained', 'fgsm', 'adv_trained'),
        ('pgd_model.pth', 'PGD Trained', 'pgd', 'adv_trained'),
        ('mixed_model.pth', 'Mixed Trained', 'mixed', 'adv_trained'),
        ('fgsm_model_pruned.pth', 'FGSM Pruned', 'fgsm_pruned', 'pruned'),
        ('pgd_model_pruned.pth', 'PGD Pruned', 'pgd_pruned', 'pruned'),
        ('mixed_model_pruned.pth', 'Mixed Pruned', 'mixed_pruned', 'pruned'),
    ]
    
    # Filter models if specific ones are selected
    if selected_models:
        selected_set = set(selected_models)
        models_config = [(f, n, s, g) for f, n, s, g in models_config if s in selected_set]
        if not models_config and 'ensemble' not in selected_set:
            print(f"Error: No valid models selected. Available: default, fgsm, pgd, mixed, fgsm_pruned, pgd_pruned, mixed_pruned, ensemble")
            return {}, num_pixels_values
    
    # Track perturbations from previous model for warm-start
    previous_perturbations = None
    previous_group = None
    
    # Evaluate individual models
    for model_file, name, short_name, similarity_group in models_config:
        full_path = os.path.join('models', model_file)
        
        if not os.path.exists(full_path):
            print(f"\n⊗ Skipping {name} - file not found: {full_path}")
            continue
        
        print(f"\n{'='*70}")
        print(f"Evaluating: {name}")
        if use_warm_start and previous_perturbations is not None:
            if similarity_group == previous_group:
                print(f"  [Warm-start from previous model in group: {previous_group}]")
            else:
                print(f"  [New similarity group: {similarity_group}, starting fresh]")
                previous_perturbations = None
        print(f"{'='*70}")
        
        try:
            net = Net()
            net.load(full_path)
            net.to(device)
            net.eval()
            
            # Use warm-start only if in same similarity group
            warm_start_perts = previous_perturbations if (similarity_group == previous_group) else None
            
            results, perturbations = evaluate_model_one_pixel(
                net, valid_loader, num_pixels_values, 
                max_images=max_images, pop_size=pop_size, 
                max_iter=max_iter,
                use_warm_start=use_warm_start,
                previous_model_perturbations=warm_start_perts
            )
            all_results[name] = results
            
            # Store perturbations for next model in same group
            if use_warm_start:
                previous_perturbations = perturbations
                previous_group = similarity_group
            
        except Exception as e:
            print(f"Error loading {name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Evaluate ensemble (if selected or no filter)
    if selected_models is None or 'ensemble' in selected_models:
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
                
                # Don't use warm-start for ensemble (different architecture)
                results, _ = evaluate_model_one_pixel(
                    ensemble, valid_loader, num_pixels_values,
                    max_images=max_images, pop_size=pop_size, 
                    max_iter=max_iter,
                    use_warm_start=False
                )
                all_results['Ensemble'] = results
                
            except Exception as e:
                print(f"Error with ensemble: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\n⊗ Skipping ensemble - models not found")
    
    return all_results, num_pixels_values


def plot_robustness_curves(all_results, num_pixels_values, output_dir='plots'):
    """Plot robustness curves and attack success rate for One Pixel Attack"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Accuracy vs Number of Pixels
    for model_name, results in all_results.items():
        if 'one_pixel' in results:
            ax1.plot(num_pixels_values, results['one_pixel'], marker='o', label=model_name, linewidth=2)
    
    ax1.set_xlabel('Number of Modified Pixels', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Robustness Curve: One Pixel Attack', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 100])
    ax1.set_xticks(num_pixels_values)
    
    # Plot 2: Attack Success Rate
    for model_name, results in all_results.items():
        if 'one_pixel' in results:
            success_rate = [100 - acc for acc in results['one_pixel']]
            ax2.plot(num_pixels_values, success_rate, marker='o', label=model_name, linewidth=2)
    
    ax2.set_xlabel('Number of Modified Pixels', fontsize=12)
    ax2.set_ylabel('Attack Success Rate (%)', fontsize=12)
    ax2.set_title('Attack Success Rate: One Pixel Attack', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])
    ax2.set_xticks(num_pixels_values)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'robustness_curve_one_pixel.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/robustness_curve_one_pixel.png")
    plt.close()


def plot_accuracy_comparison(all_results, num_pixels_values, output_dir='plots'):
    """Plot bar chart comparing accuracies at fixed number of pixels"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Use middle pixel count for comparison
    pixels_idx = len(num_pixels_values) // 2
    num_pixels = num_pixels_values[pixels_idx]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = list(all_results.keys())
    natural_acc = [all_results[m]['natural'][pixels_idx] for m in models]
    one_pixel_acc = [all_results[m]['one_pixel'][pixels_idx] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    ax.bar(x - width/2, natural_acc, width, label='Natural', color='#2ecc71')
    ax.bar(x + width/2, one_pixel_acc, width, label=f'One Pixel (n={num_pixels})', color='#9b59b6')
    
    ax.set_xlabel('Model / Defense', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Accuracy Comparison: One Pixel Attack (n={num_pixels} pixels)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison_bar_one_pixel.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/accuracy_comparison_bar_one_pixel.png")
    plt.close()


def plot_ablation_study(all_results, num_pixels_values, output_dir='plots'):
    """Plot defense layer ablation study for One Pixel Attack"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Define progression of defenses
    ablation_order = [
        'Default (No Defense)',
        'PGD Trained',
        'PGD Pruned',
        'Ensemble',
    ]
    
    # Filter to only models in ablation order
    ablation_results = {k: v for k, v in all_results.items() if k in ablation_order}
    
    if len(ablation_results) < 2:
        print("⊗ Not enough models for ablation study")
        return
    
    # Use middle pixel count
    pixels_idx = len(num_pixels_values) // 2
    num_pixels = num_pixels_values[pixels_idx]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = [m for m in ablation_order if m in ablation_results]
    one_pixel_acc = [ablation_results[m]['one_pixel'][pixels_idx] for m in models]
    
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(models)))
    ax.barh(models, one_pixel_acc, color=colors)
    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Defense Ablation Study: One Pixel Attack (n={num_pixels})', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 100])
    ax.grid(True, alpha=0.3, axis='x')
    
    for i, v in enumerate(one_pixel_acc):
        ax.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ablation_study_one_pixel.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/ablation_study_one_pixel.png")
    plt.close()



def save_results_table(all_results, num_pixels_values, output_dir='plots'):
    """Save results as JSON and markdown table"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Use middle pixel count
    pixels_idx = len(num_pixels_values) // 2
    num_pixels = num_pixels_values[pixels_idx]
    
    # Create markdown table
    md_content = f"# One Pixel Attack Evaluation Results\n\n"
    md_content += f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md_content += f"## Accuracy at num_pixels = {num_pixels}\n\n"
    md_content += "| Model | Natural (%) | One Pixel (%) |\n"
    md_content += "|-------|-------------|---------------|\n"
    
    for model_name, results in all_results.items():
        nat = results['natural'][pixels_idx]
        op = results['one_pixel'][pixels_idx]
        md_content += f"| {model_name} | {nat:.2f} | {op:.2f} |\n"
    
    # Save markdown with UTF-8 encoding
    with open(os.path.join(output_dir, 'results_summary_one_pixel.md'), 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"✓ Saved: {output_dir}/results_summary_one_pixel.md")
    
    # Save JSON
    results_data = {
        'num_pixels_values': num_pixels_values,
        'results': all_results,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, 'results_one_pixel.json'), 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2)
    print(f"✓ Saved: {output_dir}/results_one_pixel.json")


def generate_attack_example(model_name='default', max_attempts=100, num_pixels=1, 
                           pop_size=400, max_iter=100, output_dir='plots'):
    """
    Génère un exemple visuel d'une attaque One Pixel réussie.
    
    Args:
        model_name: Nom du modèle à attaquer ('default', 'pgd', etc.)
        max_attempts: Nombre maximum d'images à essayer
        num_pixels: Nombre de pixels à modifier
        pop_size: Taille de population pour DE
        max_iter: Itérations max pour DE
        output_dir: Dossier de sortie
    """
    import matplotlib.patches as patches
    
    print("\n" + "="*70)
    print(f"RECHERCHE D'UN EXEMPLE D'ATTAQUE RÉUSSIE")
    print(f"Modèle: {model_name}, Pixels modifiés: {num_pixels}")
    print("="*70 + "\n")
    
    # Charger le modèle
    model_files = {
        'default': 'models/default_model.pth',
        'fgsm': 'models/fgsm_model.pth',
        'pgd': 'models/pgd_model.pth',
        'mixed': 'models/mixed_model.pth',
        'fgsm_pruned': 'models/fgsm_model_pruned.pth',
        'pgd_pruned': 'models/pgd_model_pruned.pth',
        'mixed_pruned': 'models/mixed_model_pruned.pth',
    }
    
    if model_name not in model_files:
        print(f"Erreur: modèle '{model_name}' inconnu. Choix: {list(model_files.keys())}")
        return
    
    model_path = model_files[model_name]
    if not os.path.exists(model_path):
        print(f"Erreur: fichier modèle introuvable: {model_path}")
        return
    
    # Charger le modèle
    net = Net()
    net.load(model_path)
    net.to(device)
    net.eval()
    print(f"✓ Modèle chargé: {model_path}\n")
    
    # Charger CIFAR-10
    cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=transforms.ToTensor())
    
    # Classes CIFAR-10
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Chercher une attaque réussie
    print(f"Recherche d'une attaque réussie (max {max_attempts} tentatives)...")
    
    for attempt in range(max_attempts):
        # Prendre une image aléatoire
        idx = np.random.randint(0, len(cifar))
        image, true_label = cifar[idx]
        image = image.unsqueeze(0).to(device)
        true_label_tensor = torch.tensor([true_label]).to(device)
        
        # Prédiction originale
        with torch.no_grad():
            output_orig = net(image)
            pred_orig = output_orig.argmax(dim=1).item()
        
        # Vérifier que l'image est bien classifiée au départ
        if pred_orig != true_label:
            continue
        
        # Attaquer l'image
        adv_image, perturbations = one_pixel_attack(
            net, image, true_label_tensor, 
            num_pixels=num_pixels,
            pop_size=pop_size, 
            max_iter=max_iter, 
            verbose=False
        )
        
        # Prédiction après attaque
        with torch.no_grad():
            output_adv = net(adv_image)
            pred_adv = output_adv.argmax(dim=1).item()
        
        # Vérifier que l'attaque a réussi
        if pred_adv != true_label:
            print(f"✓ Attaque réussie à la tentative {attempt + 1}!")
            print(f"  - Classe réelle: {class_names[true_label]}")
            print(f"  - Prédiction originale: {class_names[pred_orig]}")
            print(f"  - Prédiction après attaque: {class_names[pred_adv]}")
            
            # Extraire les informations de perturbation
            perturbation = perturbations[0]  # Premier (et seul) élément du batch
            
            # Créer la visualisation
            _create_attack_visualization(
                image.cpu(), adv_image.cpu(), perturbation,
                true_label, pred_orig, pred_adv,
                class_names, num_pixels, model_name, output_dir
            )
            
            return True
    
    print(f"\n✗ Aucune attaque réussie trouvée après {max_attempts} tentatives.")
    print("  Essayez d'augmenter max_attempts ou de changer de modèle.")
    return False


def _create_attack_visualization(image_orig, image_adv, perturbation, 
                                 true_label, pred_orig, pred_adv,
                                 class_names, num_pixels, model_name, output_dir):
    """
    Crée la visualisation de l'attaque avec 3 sous-images.
    """
    import matplotlib.patches as patches
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Convertir les tenseurs en numpy pour affichage
    img_orig_np = image_orig.squeeze(0).permute(1, 2, 0).numpy()
    img_adv_np = image_adv.squeeze(0).permute(1, 2, 0).numpy()
    
    # Créer l'image de perturbation (fond noir + pixels modifiés)
    img_pert_np = np.zeros_like(img_orig_np)
    
    # Extraire les coordonnées et couleurs des pixels modifiés
    modified_pixels = []
    for i in range(num_pixels):
        x = int(perturbation[i*5].item())
        y = int(perturbation[i*5 + 1].item())
        r = perturbation[i*5 + 2].item()
        g = perturbation[i*5 + 3].item()
        b = perturbation[i*5 + 4].item()
        
        img_pert_np[y, x] = [r, g, b]
        modified_pixels.append((x, y, r, g, b))
    
    # Créer la figure avec 3 sous-graphiques
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Image originale
    axes[0].imshow(img_orig_np)
    axes[0].set_title(f'Image Originale\n'
                     f'Vraie classe: {class_names[true_label]}\n'
                     f'Prédiction: {class_names[pred_orig]} ✓',
                     fontsize=12, fontweight='bold', color='green')
    axes[0].axis('off')
    
    # Marquer les pixels qui seront modifiés
    for x, y, _, _, _ in modified_pixels:
        rect = patches.Rectangle((x-0.5, y-0.5), 1, 1, 
                                 linewidth=2, edgecolor='red', 
                                 facecolor='none')
        axes[0].add_patch(rect)
    
    # Image de perturbation (fond noir + pixel modifié visible)
    axes[1].imshow(img_pert_np)
    axes[1].set_title(f'Perturbation\n({num_pixels} pixel{"s" if num_pixels > 1 else ""} modifié{"s" if num_pixels > 1 else ""})',
                     fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Afficher les coordonnées et couleurs
    info_text = ""
    for i, (x, y, r, g, b) in enumerate(modified_pixels, 1):
        info_text += f"Pixel {i}: ({x}, {y})\nRGB: ({r:.2f}, {g:.2f}, {b:.2f})\n"
    axes[1].text(0.5, -0.15, info_text, 
                transform=axes[1].transAxes,
                fontsize=9, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Image adversariale
    axes[2].imshow(img_adv_np)
    axes[2].set_title(f'Image Attaquée\n'
                     f'Vraie classe: {class_names[true_label]}\n'
                     f'Prédiction: {class_names[pred_adv]} ✗',
                     fontsize=12, fontweight='bold', color='red')
    axes[2].axis('off')
    
    # Marquer les pixels modifiés
    for x, y, _, _, _ in modified_pixels:
        rect = patches.Rectangle((x-0.5, y-0.5), 1, 1, 
                                 linewidth=2, edgecolor='lime', 
                                 facecolor='none')
        axes[2].add_patch(rect)
    
    plt.suptitle(f'Exemple d\'Attaque One Pixel Réussie (Modèle: {model_name})',
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Sauvegarder
    filename = f'attack_example_{model_name}_{num_pixels}pixel.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualisation sauvegardée: {filepath}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate One Pixel Attack evaluation plots")
    parser.add_argument('--all', action='store_true', help="Generate all plots (default)")
    parser.add_argument('--quick', action='store_true', help="Quick evaluation with fewer pixel values and images")
    parser.add_argument('--plot-only', action='store_true', help="Only generate plots from existing results_one_pixel.json")
    parser.add_argument('--output-dir', default='plots', help="Output directory for plots")
    parser.add_argument('--max-images', type=int, default=None, help="Maximum number of images to test (default: 500 for --all, 100 for --quick)")
    parser.add_argument('--pop-size', type=int, default=400, help="Population size for DE (default: 400)")
    parser.add_argument('--max-iter', type=int, default=100, help="Max iterations for DE (default: 100)")
    parser.add_argument('--models', nargs='+', help="Specific models to evaluate (e.g., --models default pgd ensemble). Available: default, fgsm, pgd, mixed, fgsm_pruned, pgd_pruned, mixed_pruned, ensemble")
    parser.add_argument('--no-warm-start', action='store_true', help="Disable warm-start optimization (attacks won't be transferred)")
    parser.add_argument('--generate-example', action='store_true', 
                        help="Générer un exemple d'attaque One Pixel réussie")
    parser.add_argument('--example-model', type=str, default='default',
                        help="Modèle à utiliser pour l'exemple (default: 'default')")
    parser.add_argument('--example-pixels', type=int, default=1,
                        help="Nombre de pixels pour l'exemple (default: 1)")
    parser.add_argument('--max-attempts', type=int, default=100,
                        help="Nombre max d'images à essayer pour trouver une attaque réussie (default: 100)")

    args = parser.parse_args()

    if args.generate_example:
        generate_attack_example(
            model_name=args.example_model,
            max_attempts=args.max_attempts,
            num_pixels=args.example_pixels,
            pop_size=args.pop_size,
            max_iter=args.max_iter,
            output_dir=args.output_dir
        )
        return
        
    # Default to --all if nothing specified
    if not any([args.all, args.quick, args.plot_only]):
        args.all = True
    
    # Define pixel count values for evaluation
    if args.quick:
        num_pixels_values = [1, 3, 5]
        if args.max_images is None:
            args.max_images = 100  # Quick mode defaults to 100 images
    else:
        num_pixels_values = [1, 2, 3, 4, 5]
        if args.max_images is None:
            args.max_images = 500  # Standard mode defaults to 500 images
    
    # Run evaluation or load results
    if args.plot_only:
        results_file = os.path.join(args.output_dir, 'results_one_pixel.json')
        if not os.path.exists(results_file):
            print(f"Error: {results_file} not found. Run evaluation first.")
            return
        
        with open(results_file, 'r') as f:
            data = json.load(f)
            all_results = data['results']
            num_pixels_values = data['num_pixels_values']
        print(f"✓ Loaded results from {results_file}")
    else:
        all_results, num_pixels_values = run_evaluation(
            num_pixels_values, quick=args.quick, 
            max_images=args.max_images,
            pop_size=args.pop_size,
            max_iter=args.max_iter,
            selected_models=args.models,
            use_warm_start=not args.no_warm_start
        )
    
    # Generate all plots
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70 + "\n")
    
    plot_robustness_curves(all_results, num_pixels_values, args.output_dir)
    plot_accuracy_comparison(all_results, num_pixels_values, args.output_dir)
    plot_ablation_study(all_results, num_pixels_values, args.output_dir)
    
    # Save results
    if not args.plot_only:
        save_results_table(all_results, num_pixels_values, args.output_dir)
    
    print("\n" + "="*70)
    print("✓ ALL PLOTS GENERATED SUCCESSFULLY!")
    print("="*70)
    print(f"\nPlots saved to: {args.output_dir}/")
    print("\nGenerated files:")
    print("  - robustness_curve_one_pixel.png")
    print("  - accuracy_comparison_bar_one_pixel.png")
    print("  - ablation_study_one_pixel.png")
    print("  - results_summary_one_pixel.md")
    print("  - results_one_pixel.json")


if __name__ == "__main__":
    main()