#!/usr/bin/env python3 
import os
import argparse
import random

import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from tqdm import tqdm

USE_RANDOMIZATION = True

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

valid_size = 1024 
batch_size = 32 

def random_resize(images, resize_min=28, resize_max=32):
    """
    Redimensionnement aléatoire des images (CIFAR: 32x32 -> taille aléatoire entre 28 et 32).
    images: tensor (B, C, H, W)
    """
    size = random.randint(resize_min, resize_max)
    return F.interpolate(images, size=(size, size), mode='bilinear', align_corners=False)

def random_pad(images, final_size=32):
    """
    Padding aléatoire pour revenir à une taille finale (par défaut 32x32).
    """
    current_size = images.size(-1)
    diff = final_size - current_size

    # Si déjà à la taille finale, pas de padding
    if diff <= 0:
        return images

    pad_top = random.randint(0, diff)
    pad_left = random.randint(0, diff)
    pad_bottom = diff - pad_top
    pad_right = diff - pad_left

    # F.pad: (left, right, top, bottom)
    return F.pad(images, (pad_left, pad_right, pad_top, pad_bottom))

def randomization_layer(images, resize_min=28, resize_max=32, final_size=32):
    """
    Combine random resize + random padding.
    À appeler juste avant de passer les images au réseau.
    """
    x = random_resize(images, resize_min, resize_max)
    x = random_pad(x, final_size)
    return x

'''Basic neural network architecture (from pytorch doc).'''
class Net(nn.Module):

    model_file="models/mixed_model.pth"
    '''This file will be loaded to test your model. Use --model-file to load/store a different model.'''

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x

    def save(self, model_file):
        '''Helper function, use it to save the model weights after training.'''
        torch.save(self.state_dict(), model_file)

    def load(self, model_file):
        self.load_state_dict(torch.load(model_file, map_location=torch.device(device)))

        
    def load_for_testing(self, project_dir='./'):
        '''This function will be called automatically before testing your
           project, and will load the model weights from the file
           specify in Net.model_file.
           
           You must not change the prototype of this function. You may
           add extra code in its body if you feel it is necessary, but
           beware that paths of files used in this function should be
           refered relative to the root of your project directory.
        '''        
        self.load(os.path.join(project_dir, Net.model_file))


############################################
#  Pruning Functions
############################################

def prune_model(net, sparsity=0.5):
    '''
    Prune model weights using magnitude-based pruning.
    
    Args:
        net: Neural network model
        sparsity: Percentage of weights to prune (0.0 to 1.0)
    
    Returns:
        Pruned model
    '''
    import torch.nn.utils.prune as prune
    
    print(f"Pruning model with {sparsity*100:.0f}% sparsity...")
    
    # Prune convolutional and linear layers
    parameters_to_prune = []
    for name, module in net.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))
    
    # Apply global unstructured pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=sparsity,
    )
    
    # Make pruning permanent
    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)
    
    print("Pruning complete!")
    return net


def fine_tune_pruned_model(net, train_loader, num_epochs=8):
    '''
    Fine-tune a pruned model to recover accuracy.
    Uses adaptive learning rate schedule for better recovery.
    '''
    print(f"Fine-tuning pruned model for {num_epochs} epochs...")
    criterion = nn.NLLLoss()
    
    # Start with slightly higher LR, will decay
    optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9, weight_decay=1e-4)
    
    # Learning rate scheduler for better convergence
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 500 == 499:
                print('[%d, %5d] loss: %.3f, lr: %.6f' % (epoch + 1, i + 1, running_loss / 500, optimizer.param_groups[0]['lr']))
                running_loss = 0.0
        
        # Step the learning rate scheduler
        scheduler.step()
        print(f'Epoch {epoch + 1}/{num_epochs} complete, LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    print("Fine-tuning complete!")
    return net


############################################
#  Ensemble Model
############################################

class EnsembleNet(nn.Module):
    '''
    Ensemble of multiple models for robust prediction.
    Combines predictions via weighted soft voting based on model performance.
    
    Weights are based on empirical PGD robustness on validation set:
    - PGD model: Best overall robustness (~25% PGD accuracy) → weight 0.45
    - Mixed model: Good generalization (~23% PGD accuracy) → weight 0.35
    - FGSM model: Weaker against PGD (~21% PGD accuracy) → weight 0.20
    '''
    
    model_file = "models/ensemble_model.pth"  # Not actually used, but needed for compatibility
    
    def __init__(self, model_files=None, weights=None):
        super().__init__()
        if model_files is None:
            model_files = [
                'models/fgsm_model.pth',
                'models/pgd_model.pth', 
                'models/mixed_model.pth'
            ]
        
        # Default weights based on validation performance
        # Order: FGSM, PGD, Mixed
        if weights is None:
            weights = [0.20, 0.45, 0.35]  # PGD gets highest weight
        
        self.model_files = model_files
        self.models = nn.ModuleList()
        self.weights = torch.tensor(weights, dtype=torch.float32, device=device)
        
        # Normalize weights to sum to 1
        self.weights = self.weights / self.weights.sum()
        
        # Load all models
        for model_file in model_files:
            if os.path.exists(model_file):
                model = Net()
                model.load(model_file)
                model.eval()  # Set to evaluation mode
                self.models.append(model)
                print(f"Loaded model: {model_file}")
            else:
                print(f"Warning: Model file not found: {model_file}")
        
        print(f"Ensemble weights: {self.weights.cpu().numpy()}")
    
    def forward(self, x):
        '''
        Forward pass through ensemble - weighted average of predictions.
        '''
        if len(self.models) == 0:
            raise RuntimeError("No models loaded in ensemble!")
        
        # Get predictions from all models
        outputs = []
        for model in self.models:
            out = model(x)
            outputs.append(out)
        
        # Weighted average of log probabilities
        outputs_tensor = torch.stack(outputs)  # Shape: (num_models, batch_size, num_classes)
        weights_expanded = self.weights.view(-1, 1, 1)  # Shape: (num_models, 1, 1)
        ensemble_output = (outputs_tensor * weights_expanded).sum(dim=0)
        
        return ensemble_output
    
    def save(self, model_file):
        '''Save ensemble configuration (model files list)'''
        torch.save({'model_files': self.model_files}, model_file)
        print(f"Ensemble configuration saved to {model_file}")
    
    def load(self, model_file):
        '''Load ensemble configuration'''
        config = torch.load(model_file, map_location=torch.device(device))
        self.model_files = config['model_files']
        print(f"Ensemble configuration loaded from {model_file}")
    
    def load_for_testing(self, project_dir='./'):
        '''Compatibility method for testing'''
        # Ensemble models are already loaded in __init__
        pass


def train_model(net, train_loader, pth_filename, num_epochs):
    '''Basic training function (from pytorch doc.)'''
    print("Starting training")
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    net.save(pth_filename)
    print('Model saved in {}'.format(pth_filename))



def fgsm_attack(model, images, labels, epsilon):
    '''
    Fast Gradient Sign Method (FGSM) attack.
    '''
    images = images.clone().detach()
    images.requires_grad = True
    
    # Forward pass
    outputs = model(images)
    loss = F.nll_loss(outputs, labels)
    
    # Backward pass to get gradients
    model.zero_grad()
    loss.backward()
    
    # Create adversarial examples
    data_grad = images.grad.data
    perturbed_images = images + epsilon * data_grad.sign()
    perturbed_images = torch.clamp(perturbed_images, 0, 1)
    
    return perturbed_images.detach()


def pgd_attack(model, images, labels, epsilon, alpha, num_iter):
    '''
    Projected Gradient Descent (PGD) attack.
    '''
    # Start with random perturbation
    perturbed_images = images.clone().detach()
    perturbed_images = perturbed_images + torch.empty_like(perturbed_images).uniform_(-epsilon, epsilon)
    perturbed_images = torch.clamp(perturbed_images, 0, 1)
    
    for i in range(num_iter):
        perturbed_images.requires_grad = True
        
        # Forward pass
        outputs = model(perturbed_images)
        loss = F.nll_loss(outputs, labels)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Update perturbation
        data_grad = perturbed_images.grad.data
        perturbed_images = perturbed_images.detach() + alpha * data_grad.sign()
        
        # Project back to epsilon ball
        perturbation = torch.clamp(perturbed_images - images, -epsilon, epsilon)
        perturbed_images = torch.clamp(images + perturbation, 0, 1).detach()
    
    return perturbed_images

def one_pixel_attack(model, images, labels, num_pixels=1, pop_size=400, max_iter=100, 
                     verbose=True, initial_perturbations=None):
    '''
    One Pixel Attack using Differential Evolution (GPU-optimized).
    Paper: https://arxiv.org/abs/1710.08864
    
    Args:
        model: The target model
        images: Input images (batch)
        labels: True labels
        num_pixels: Number of pixels to modify (default: 1)
        pop_size: Population size for DE
        max_iter: Maximum iterations for DE
        verbose: Show progress bar (default: True)
        initial_perturbations: Optional dict mapping image indices to previous perturbations
                              for warm starting the attack. Can be from:
                              - A previous model (same num_pixels)
                              - A previous num_pixels attack (e.g., 5 pixels → 4 pixels)
                              Format: {img_idx: perturbation_tensor}
    
    Returns:
        perturbed_images: The attacked images
        perturbations_dict: Dict mapping image indices to their final perturbations (for reuse)
    '''
    device = images.device
    batch_size = images.size(0)
    channels = images.size(1)
    height = images.size(2)
    width = images.size(3)
    
    perturbed_images = images.clone()
    perturbations_dict = {}  # Store perturbations for potential reuse
    
    # Progress bar for images
    image_iterator = tqdm(range(batch_size), desc="One Pixel Attack", disable=not verbose) if verbose else range(batch_size)
    
    # Process each image individually
    for img_idx in image_iterator:
        image = images[img_idx:img_idx+1]
        true_label = labels[img_idx]
        
        # Differential Evolution optimization
        dim = 5 * num_pixels  # (x, y, r, g, b) per pixel
        
        # Initialize population on GPU
        population = torch.rand(pop_size, dim, device=device)
        
        # Warm start: if we have initial perturbation, use it for part of population
        if initial_perturbations is not None and img_idx in initial_perturbations:
            init_pert = initial_perturbations[img_idx].to(device)
            
            # Determine if this is from a higher num_pixels (e.g., 5→4) or same model
            init_dim = init_pert.shape[0]
            init_num_pixels = init_dim // 5
            
            # Use previous solution for a portion of the population (20-30%)
            num_warm_start = max(1, pop_size // 4)
            
            if init_num_pixels >= num_pixels:
                # Case 1: Warm-start from higher num_pixels (e.g., 5→4, 4→3)
                # Take the first num_pixels from the previous solution
                for i in range(num_warm_start):
                    # Extract first num_pixels worth of parameters
                    noise_scale = 0.05 * (i / num_warm_start)  # Small noise
                    population[i, :dim] = init_pert[:dim] + torch.randn(dim, device=device) * noise_scale
                    population[i] = torch.clamp(population[i], 0, 1)
                
                if verbose and isinstance(image_iterator, tqdm):
                    image_iterator.set_postfix_str(f"Warm-start: {init_num_pixels}→{num_pixels} pixels")
            
            elif init_num_pixels < num_pixels:
                # Case 2: Warm-start from lower num_pixels (shouldn't happen with our strategy)
                # Use the existing pixels and randomly initialize the rest
                for i in range(num_warm_start):
                    noise_scale = 0.1 * (i / num_warm_start)
                    population[i, :init_dim] = init_pert + torch.randn(init_dim, device=device) * noise_scale
                    # Rest stays random
                    population[i] = torch.clamp(population[i], 0, 1)
            
            else:
                # Case 3: Same num_pixels, just add noise for diversity
                for i in range(num_warm_start):
                    noise_scale = 0.1 * (i / num_warm_start)
                    population[i] = init_pert + torch.randn_like(init_pert) * noise_scale
                    population[i] = torch.clamp(population[i], 0, 1)
        
        # Initialize coordinates properly for remaining population
        for i in range(num_pixels):
            if initial_perturbations is None or img_idx not in initial_perturbations:
                # Fully random initialization
                population[:, i*5] *= width      # x coordinate
                population[:, i*5 + 1] *= height  # y coordinate
            else:
                # Only initialize the random part
                num_warm_start = max(1, pop_size // 4)
                population[num_warm_start:, i*5] *= width
                population[num_warm_start:, i*5 + 1] *= height
            # RGB values already in [0, 1]
        
        # Evaluate fitness for initial population (vectorized)
        fitness = evaluate_fitness_batch(model, image, true_label, population, 
                                        num_pixels, channels, device)
        
        # Differential Evolution iterations
        F = 0.5   # Differential weight
        CR = 0.9  # Crossover probability
        
        for iteration in range(max_iter):
            # Vectorized mutation for entire population
            perm_a = torch.randperm(pop_size, device=device)
            perm_b = torch.randperm(pop_size, device=device)
            perm_c = torch.randperm(pop_size, device=device)
            
            a = population[perm_a]
            b = population[perm_b]
            c = population[perm_c]
            
            # Mutation: mutant = a + F * (b - c)
            mutant = a + F * (b - c)
            mutant = torch.clamp(mutant, 0, 1)
            
            # Clip coordinates
            for i in range(num_pixels):
                mutant[:, i*5] = torch.clamp(mutant[:, i*5], 0, width - 1)
                mutant[:, i*5 + 1] = torch.clamp(mutant[:, i*5 + 1], 0, height - 1)
            
            # Crossover
            cross_points = torch.rand(pop_size, dim, device=device) < CR
            rand_dims = torch.randint(0, dim, (pop_size,), device=device)
            cross_points[torch.arange(pop_size), rand_dims] = True
            
            trial = torch.where(cross_points, mutant, population)
            
            # Selection: evaluate all trials in parallel
            trial_fitness = evaluate_fitness_batch(model, image, true_label, trial,
                                                  num_pixels, channels, device)
            
            # Update population where trial is better
            improvement_mask = trial_fitness < fitness
            population[improvement_mask] = trial[improvement_mask]
            fitness[improvement_mask] = trial_fitness[improvement_mask]
        
        # Apply best perturbation
        best_idx = torch.argmin(fitness)
        best_solution = population[best_idx]
        perturbed_img = apply_perturbation(image.clone(), best_solution, 
                                          num_pixels, channels)
        perturbed_images[img_idx] = perturbed_img.squeeze(0)
        
        # Store the best perturbation for potential reuse
        perturbations_dict[img_idx] = best_solution.cpu()
    
    return perturbed_images.detach(), perturbations_dict


def evaluate_fitness_batch(model, image, true_label, population, num_pixels, channels, device):
    '''
    Vectorized fitness evaluation for entire population.
    Returns fitness for all solutions in parallel.
    '''
    pop_size = population.size(0)
    batch_images = image.repeat(pop_size, 1, 1, 1)  # Replicate image for entire population
    
    # Apply all perturbations in parallel
    for i in range(num_pixels):
        x = population[:, i*5].long()
        y = population[:, i*5 + 1].long()
        
        if channels == 3:  # RGB
            r = population[:, i*5 + 2]
            g = population[:, i*5 + 3]
            b = population[:, i*5 + 4]
            
            # Use advanced indexing to modify all images at once
            batch_idx = torch.arange(pop_size, device=device)
            batch_images[batch_idx, 0, y, x] = r
            batch_images[batch_idx, 1, y, x] = g
            batch_images[batch_idx, 2, y, x] = b
        else:  # Grayscale
            val = population[:, i*5 + 2]
            batch_idx = torch.arange(pop_size, device=device)
            batch_images[batch_idx, 0, y, x] = val
    
    batch_images = torch.clamp(batch_images, 0, 1)
    
    # Forward pass for entire batch
    with torch.no_grad():
        outputs = model(batch_images)
        probabilities = F.softmax(outputs, dim=1)
        true_class_probs = probabilities[:, true_label]
    
    return true_class_probs


def apply_perturbation(image, solution, num_pixels, channels):
    '''
    Apply pixel perturbations encoded in solution to the image.
    '''
    perturbed = image.clone()
    
    for i in range(num_pixels):
        x = int(solution[i*5].item())
        y = int(solution[i*5 + 1].item())
        
        if channels == 3:  # RGB
            perturbed[0, 0, y, x] = solution[i*5 + 2]
            perturbed[0, 1, y, x] = solution[i*5 + 3]
            perturbed[0, 2, y, x] = solution[i*5 + 4]
        else:  # Grayscale
            perturbed[0, 0, y, x] = solution[i*5 + 2]
    
    return torch.clamp(perturbed, 0, 1)


def train_adversarial(net, train_loader, pth_filename, num_epochs, adv_training_ratio=0.5, epsilon=0.03, attack_method='pgd', pgd_alpha=0.01, pgd_iters=7):
    '''
    Adversarial training function.
    '''
    print(f"Starting adversarial training with {attack_method.upper()} attack")
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            
            # Mix clean and adversarial examples
            if np.random.random() < adv_training_ratio:
                # Generate adversarial examples
                net.eval()  # Set to eval mode for attack generation
                
                if attack_method.lower() == 'fgsm':
                    adv_inputs = fgsm_attack(net, inputs, labels, epsilon)
                elif attack_method.lower() == 'pgd':
                    adv_inputs = pgd_attack(net, inputs, labels, epsilon, pgd_alpha, pgd_iters)
                elif attack_method.lower() == 'mixed':
                    # Mix both FGSM and PGD attacks
                    if np.random.random() < 0.5:
                        adv_inputs = fgsm_attack(net, inputs, labels, epsilon)
                    else:
                        adv_inputs = pgd_attack(net, inputs, labels, epsilon, pgd_alpha, pgd_iters)
                else:
                    raise ValueError(f"Unknown attack method: {attack_method}. Use 'fgsm', 'pgd', or 'mixed'.")
                
                net.train()  # Back to train mode
                outputs = net(adv_inputs)
            else:
                # Use clean examples
                outputs = net(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 500 == 499:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0

    net.save(pth_filename)
    print('Adversarially trained model saved in {}'.format(pth_filename))


def test_natural(net, test_loader):
    '''Basic testing function.'''

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i,data in enumerate(test_loader, 0):
            images, labels = data[0].to(device), data[1].to(device)

            # Apply randomization defense if enabled
            if USE_RANDOMIZATION:
                images = randomization_layer(images)

            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def test_fgsm(net, test_loader, epsilon):
    '''
    Test model robustness against FGSM attack.
    '''
    correct = 0
    total = 0
    
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        
        # Generate adversarial examples
        adv_images = fgsm_attack(net, images, labels, epsilon)

        # Apply randomization defense if enabled
        if USE_RANDOMIZATION:
            adv_images = randomization_layer(adv_images)
        
        # Test on adversarial examples
        with torch.no_grad():
            outputs = net(adv_images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total


def test_pgd(net, test_loader, epsilon, alpha, num_iter):
    '''
    Test model robustness against PGD attack.
    '''
    correct = 0
    total = 0
    
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        
        # Generate adversarial examples
        adv_images = pgd_attack(net, images, labels, epsilon, alpha, num_iter)
        
        # Apply randomization defense if enabled
        if USE_RANDOMIZATION:
            adv_images = randomization_layer(adv_images)

        # Test on adversarial examples
        with torch.no_grad():
            outputs = net(adv_images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total


def test_one_pixel(net, test_loader, num_pixels=1, pop_size=400, max_iter=100):
    '''
    Test model robustness against One Pixel attack.
    
    Args:
        net: The model to test
        test_loader: Test data loader
        num_pixels: Number of pixels to modify (default: 1)
        pop_size: Population size for DE (default: 400)
        max_iter: Maximum iterations for DE (default: 100)
    '''
    correct = 0
    total = 0
    
    # Progress bar for batches
    batch_iterator = tqdm(test_loader, desc="Testing One Pixel Attack")
    
    for data in batch_iterator:
        images, labels = data[0].to(device), data[1].to(device)
        
        # Generate adversarial examples (with nested progress bar disabled)
        adv_images = one_pixel_attack(net, images, labels, num_pixels, pop_size, max_iter, verbose=False)

        # Apply randomization defense if enabled
        if USE_RANDOMIZATION:
            adv_images = randomization_layer(adv_images)
        
        # Test on adversarial examples
        with torch.no_grad():
            outputs = net(adv_images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Update progress bar with running accuracy
        current_acc = 100 * correct / total
        batch_iterator.set_postfix({'accuracy': f'{current_acc:.2f}%'})
    
    return 100 * correct / total




def get_train_loader(dataset, valid_size=1024, batch_size=32):
    '''Split dataset into [train:valid] and return a DataLoader for the training part.'''

    indices = list(range(len(dataset)))
    train_sampler = torch.utils.data.SubsetRandomSampler(indices[valid_size:])
    train = torch.utils.data.DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)

    return train

def get_validation_loader(dataset, valid_size=1024, batch_size=32):
    '''Split dataset into [train:valid] and return a DataLoader for the validation part.'''

    indices = list(range(len(dataset)))
    valid_sampler = torch.utils.data.SubsetRandomSampler(indices[:valid_size])
    valid = torch.utils.data.DataLoader(dataset, sampler=valid_sampler, batch_size=batch_size)

    return valid



# def main():

#     #### Parse command line arguments 
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model-file", default=Net.model_file,
#                         help="Name of the file used to load or to sore the model weights."
#                         "If the file exists, the weights will be load from it."
#                         "If the file doesn't exists, or if --force-train is set, training will be performed, "
#                         "and the model weights will be stored in this file."
#                         "Warning: "+Net.model_file+" will be used for testing (see load_for_testing()).")
#     parser.add_argument('-f', '--force-train', action="store_true",
#                         help="Force training even if model file already exists"
#                              "Warning: previous model file will be erased!).")
#     parser.add_argument('-e', '--num-epochs', type=int, default=10,
#                         help="Set the number of epochs during training")
#     parser.add_argument('--adv-train', action="store_true",
#                         help="Use adversarial training instead of standard training")
#     parser.add_argument('--attack-method', type=str, default='pgd', choices=['fgsm', 'pgd', 'mixed'],
#                         help="Attack method to use for adversarial training: 'fgsm', 'pgd', or 'mixed' (default: pgd)")
#     parser.add_argument('--adv-ratio', type=float, default=0.5,
#                         help="Ratio of adversarial examples in adversarial training (default: 0.5)")
#     parser.add_argument('--epsilon', type=float, default=0.03,
#                         help="Epsilon for adversarial attacks (default: 0.03)")
#     parser.add_argument('--test-attacks', action="store_true",
#                         help="Test the model against FGSM and PGD attacks")
#     parser.add_argument('--randomization', action="store_true",
#                         help="Enable randomization defense (Xie et al. 2018) at test time")
#     parser.add_argument('--prune', action="store_true",
#                         help="Prune the model after loading")
#     parser.add_argument('--sparsity', type=float, default=0.3,
#                         help="Sparsity level for pruning (0.0-1.0, default: 0.3)")
#     parser.add_argument('--fine-tune-epochs', type=int, default=8,
#                         help="Number of epochs for fine-tuning after pruning (default: 8)")
#     parser.add_argument('--use-ensemble', action="store_true",
#                         help="Use ensemble of models instead of single model")
#     parser.add_argument('--ensemble-models', nargs='+',
#                         help="List of model files for ensemble (default: pruned_fgsm, pruned_pgd, pruned_mixed)")
#     args = parser.parse_args()

#     # Set global flag for randomization
#     global USE_RANDOMIZATION
#     USE_RANDOMIZATION = args.randomization

#     if USE_RANDOMIZATION:
#         print("Randomization defense ENABLED at test time.")
#     else:
#         print("Randomization defense DISABLED (standard evaluation).")

#     #### Create model and move it to whatever device is available (gpu/cpu)
#     if args.use_ensemble:
#         print("Using ENSEMBLE model")
#         net = EnsembleNet(model_files=args.ensemble_models)
#         net.to(device)
#     else:
#         net = Net()
#         net.to(device)

#     #### Model training (if necessary)
#     if not args.use_ensemble and (not os.path.exists(args.model_file) or args.force_train):
#         print("Training model")
#         print(args.model_file)

#         train_transform = transforms.Compose([transforms.ToTensor()]) 
#         cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=train_transform)
#         train_loader = get_train_loader(cifar, valid_size, batch_size=batch_size)
        
#         if args.adv_train:
#             train_adversarial(net, train_loader, args.model_file, args.num_epochs, 
#                             adv_training_ratio=args.adv_ratio, epsilon=args.epsilon,
#                             attack_method=args.attack_method)
#         else:
#             train_model(net, train_loader, args.model_file, args.num_epochs)
#         print("Model save to '{}'.".format(args.model_file))
    
#     #### Pruning (if requested)
#     if args.prune and not args.use_ensemble:
#         print("\n" + "="*50)
#         print("PRUNING MODEL")
#         print("="*50)
        
#         # Load model if not just trained
#         if os.path.exists(args.model_file):
#             net.load(args.model_file)
        
#         # Prune the model
#         net = prune_model(net, sparsity=args.sparsity)
        
#         # Fine-tune if requested
#         if args.fine_tune_epochs > 0:
#             train_transform = transforms.Compose([transforms.ToTensor()])
#             cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=train_transform)
#             train_loader = get_train_loader(cifar, valid_size, batch_size=batch_size)
#             net = fine_tune_pruned_model(net, train_loader, num_epochs=args.fine_tune_epochs)
        
#         # Save pruned model
#         pruned_file = args.model_file.replace('.pth', '_pruned.pth')
#         net.save(pruned_file)
#         print(f"Pruned model saved to: {pruned_file}")

#     #### Model testing
#     if args.use_ensemble:
#         print("\nTesting ENSEMBLE model")
#     else:
#         print("Testing with model from '{}'. ".format(args.model_file))

#     # Note: You should not change the transform applied to the
#     # validation dataset since, it will be the only transform used
#     # during final testing.
#     cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=transforms.ToTensor()) 
#     valid_loader = get_validation_loader(cifar, valid_size)

#     if not args.use_ensemble:
#         net.load(args.model_file)

#     acc = test_natural(net, valid_loader)
#     print("Model natural accuracy (valid): {:.2f}%".format(acc))
    
#     # Test against adversarial attacks
#     if args.test_attacks:
#         print("\nTesting against adversarial attacks:")
#         fgsm_acc = test_fgsm(net, valid_loader, args.epsilon)
#         print("FGSM accuracy (epsilon={}): {:.2f}%".format(args.epsilon, fgsm_acc))
        
#         pgd_acc = test_pgd(net, valid_loader, args.epsilon, alpha=0.01, num_iter=7)
#         print("PGD accuracy (epsilon={}, alpha=0.01, iters=7): {:.2f}%".format(args.epsilon, pgd_acc))

#     if args.model_file != Net.model_file:
#         print("Warning: '{0}' is not the default model file, "\
#               "it will not be the one used for testing your project. "\
#               "If this is your best model, "\
#               "you should rename/link '{0}' to '{1}'.".format(args.model_file, Net.model_file))


def main():

    #### Parse command line arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", default=Net.model_file,
                        help="Name of the file used to load or to sore the model weights."
                        "If the file exists, the weights will be load from it."
                        "If the file doesn't exists, or if --force-train is set, training will be performed, "
                        "and the model weights will be stored in this file."
                        "Warning: "+Net.model_file+" will be used for testing (see load_for_testing()).")
    parser.add_argument('-f', '--force-train', action="store_true",
                        help="Force training even if model file already exists"
                             "Warning: previous model file will be erased!).")
    parser.add_argument('-e', '--num-epochs', type=int, default=10,
                        help="Set the number of epochs during training")
    parser.add_argument('--adv-train', action="store_true",
                        help="Use adversarial training instead of standard training")
    parser.add_argument('--attack-method', type=str, default='pgd', choices=['fgsm', 'pgd', 'mixed'],
                        help="Attack method to use for adversarial training: 'fgsm', 'pgd', or 'mixed' (default: pgd)")
    parser.add_argument('--adv-ratio', type=float, default=0.5,
                        help="Ratio of adversarial examples in adversarial training (default: 0.5)")
    parser.add_argument('--epsilon', type=float, default=0.03,
                        help="Epsilon for adversarial attacks (default: 0.03)")
    parser.add_argument('--test-attacks', action="store_true",
                        help="Test the model against FGSM and PGD attacks")
    parser.add_argument('--test-one-pixel', action="store_true",
                        help="Test the model against One Pixel attack")
    parser.add_argument('--one-pixel-samples', type=int, default=50,
                        help="Number of images to test for One Pixel attack (default: 50)")
    parser.add_argument('--num-pixels', type=int, default=1,
                        help="Number of pixels to modify in One Pixel attack (default: 1)")
    parser.add_argument('--pop-size', type=int, default=400,
                        help="Population size for DE in One Pixel attack (default: 400)")
    parser.add_argument('--max-iter', type=int, default=100,
                        help="Max iterations for DE in One Pixel attack (default: 100)")
    parser.add_argument('--randomization', action="store_true",
                        help="Enable randomization defense (Xie et al. 2018) at test time")
    parser.add_argument('--prune', action="store_true",
                        help="Prune the model after loading")
    parser.add_argument('--sparsity', type=float, default=0.3,
                        help="Sparsity level for pruning (0.0-1.0, default: 0.3)")
    parser.add_argument('--fine-tune-epochs', type=int, default=8,
                        help="Number of epochs for fine-tuning after pruning (default: 8)")
    parser.add_argument('--use-ensemble', action="store_true",
                        help="Use ensemble of models instead of single model")
    parser.add_argument('--ensemble-models', nargs='+',
                        help="List of model files for ensemble (default: pruned_fgsm, pruned_pgd, pruned_mixed)")
    args = parser.parse_args()

    # Set global flag for randomization
    global USE_RANDOMIZATION
    USE_RANDOMIZATION = args.randomization

    if USE_RANDOMIZATION:
        print("Randomization defense ENABLED at test time.")
    else:
        print("Randomization defense DISABLED (standard evaluation).")

    #### Create model and move it to whatever device is available (gpu/cpu)
    if args.use_ensemble:
        print("Using ENSEMBLE model")
        net = EnsembleNet(model_files=args.ensemble_models)
        net.to(device)
    else:
        net = Net()
        net.to(device)

    #### Model training (if necessary)
    if not args.use_ensemble and (not os.path.exists(args.model_file) or args.force_train):
        print("Training model")
        print(args.model_file)

        train_transform = transforms.Compose([transforms.ToTensor()]) 
        cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=train_transform)
        train_loader = get_train_loader(cifar, valid_size, batch_size=batch_size)
        
        if args.adv_train:
            train_adversarial(net, train_loader, args.model_file, args.num_epochs, 
                            adv_training_ratio=args.adv_ratio, epsilon=args.epsilon,
                            attack_method=args.attack_method)
        else:
            train_model(net, train_loader, args.model_file, args.num_epochs)
        print("Model save to '{}'.".format(args.model_file))
    
    #### Pruning (if requested)
    if args.prune and not args.use_ensemble:
        print("\n" + "="*50)
        print("PRUNING MODEL")
        print("="*50)
        
        # Load model if not just trained
        if os.path.exists(args.model_file):
            net.load(args.model_file)
        
        # Prune the model
        net = prune_model(net, sparsity=args.sparsity)
        
        # Fine-tune if requested
        if args.fine_tune_epochs > 0:
            train_transform = transforms.Compose([transforms.ToTensor()])
            cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=train_transform)
            train_loader = get_train_loader(cifar, valid_size, batch_size=batch_size)
            net = fine_tune_pruned_model(net, train_loader, num_epochs=args.fine_tune_epochs)
        
        # Save pruned model
        pruned_file = args.model_file.replace('.pth', '_pruned.pth')
        net.save(pruned_file)
        print(f"Pruned model saved to: {pruned_file}")

    #### Model testing
    if args.use_ensemble:
        print("\nTesting ENSEMBLE model")
    else:
        print("Testing with model from '{}'. ".format(args.model_file))

    # Note: You should not change the transform applied to the
    # validation dataset since, it will be the only transform used
    # during final testing.
    cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=transforms.ToTensor()) 
    valid_loader = get_validation_loader(cifar, valid_size)

    if not args.use_ensemble:
        net.load(args.model_file)

    acc = test_natural(net, valid_loader)
    print("Model natural accuracy (valid): {:.2f}%".format(acc))
    
    # Test against adversarial attacks
    if args.test_attacks:
        print("\nTesting against adversarial attacks:")
        fgsm_acc = test_fgsm(net, valid_loader, args.epsilon)
        print("FGSM accuracy (epsilon={}): {:.2f}%".format(args.epsilon, fgsm_acc))
        
        pgd_acc = test_pgd(net, valid_loader, args.epsilon, alpha=0.01, num_iter=7)
        print("PGD accuracy (epsilon={}, alpha=0.01, iters=7): {:.2f}%".format(args.epsilon, pgd_acc))
    
    # Test against One Pixel attack
    if args.test_one_pixel:
        print("\n" + "="*50)
        print("Testing against One Pixel attack")
        print("="*50)
        
        # Create a random subset of validation data
        import random
        from torch.utils.data import Subset
        
        total_valid_samples = len(cifar) - valid_size  # Adjust based on your valid_size
        # Get indices for validation set
        valid_indices = list(range(len(cifar) - valid_size, len(cifar)))
        num_samples = min(args.one_pixel_samples, len(valid_indices))
        
        # Set seed for reproducibility
        random.seed(42)
        random_indices = random.sample(valid_indices, num_samples)
        subset_dataset = Subset(cifar, random_indices)
        subset_loader = torch.utils.data.DataLoader(subset_dataset, batch_size=32, shuffle=False)
        
        print(f"Testing on {num_samples} randomly selected validation images")
        print(f"Parameters: num_pixels={args.num_pixels}, pop_size={args.pop_size}, max_iter={args.max_iter}")
        
        one_pixel_acc = test_one_pixel(net, subset_loader,
                                        num_pixels=args.num_pixels,
                                        pop_size=args.pop_size,
                                        max_iter=args.max_iter)
        print(f"\nOne Pixel Attack accuracy: {one_pixel_acc:.2f}%")
        print(f"Attack success rate: {100 - one_pixel_acc:.2f}%")

    if args.model_file != Net.model_file:
        print("Warning: '{0}' is not the default model file, "\
              "it will not be the one used for testing your project. "\
              "If this is your best model, "\
              "you should rename/link '{0}' to '{1}'.".format(args.model_file, Net.model_file))

if __name__ == "__main__":
    main()

