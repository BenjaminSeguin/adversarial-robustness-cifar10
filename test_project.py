#!/usr/bin/env python3

import os, os.path, sys
import argparse
import importlib 
import importlib.abc
import torch, torchvision
import torchvision.transforms as transforms

torch.seed()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def load_project(project_dir):
    module_filename = os.path.join(project_dir, 'model.py')
    if os.path.exists(project_dir) and os.path.isdir(project_dir) and os.path.isfile(module_filename):
        print("Found valid project in '{}'.".format(project_dir))
    else:
        print("Fatal: '{}' is not a valid project directory.".format(project_dir))
        raise FileNotFoundError 

    sys.path = [project_dir] + sys.path
    spec = importlib.util.spec_from_file_location("model", module_filename)
    project_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(project_module)

    return project_module

def test_natural(net, test_loader, num_samples, use_randomization=False, randomization_fn=None):
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            batch_size = labels.size(0)
            
            # Collect predictions across multiple samples
            if num_samples > 1 and use_randomization:
                # Average probabilities across multiple randomized samples
                accumulated_outputs = None
                for _ in range(num_samples):
                    test_images = randomization_fn(images) if randomization_fn is not None else images
                    outputs = net(test_images)
                    
                    if accumulated_outputs is None:
                        accumulated_outputs = outputs
                    else:
                        accumulated_outputs += outputs
                
                # Average and get final prediction
                avg_outputs = accumulated_outputs / num_samples
                _, predicted = torch.max(avg_outputs.data, 1)
            else:
                # Single sample - no averaging needed
                test_images = images
                if use_randomization and randomization_fn is not None:
                    test_images = randomization_fn(images)
                outputs = net(test_images)
                _, predicted = torch.max(outputs.data, 1)
            
            total += batch_size
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

def get_validation_loader(dataset, valid_size=1024, batch_size=32):
    '''Split dataset into [train:valid] and return a DataLoader for the validation part.'''

    indices = list(range(len(dataset)))
    valid_sampler = torch.utils.data.SubsetRandomSampler(indices[:valid_size])
    valid = torch.utils.data.DataLoader(dataset, sampler=valid_sampler, batch_size=batch_size)
    return valid

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("project_dir", metavar="project-dir", nargs="?", default=os.getcwd(),
                        help="Path to the project directory to test.")
    parser.add_argument("-b", "--batch-size", type=int, default=256,
                        help="Set batch size.")
    parser.add_argument("-s", "--num-samples", type=int, default=10,
                        help="Num samples for testing (required to test randomized networks).")
    parser.add_argument("--randomization", action="store_true", default=True,
                        help="Enable randomization defense at test time")
    parser.add_argument("--ensemble", action="store_true", default=True,
                        help="Use ensemble of adversarially trained models instead of single model")

    args = parser.parse_args()
    project_module = load_project(args.project_dir)

    # Load ensemble or single model
    if args.ensemble:
        if hasattr(project_module, 'EnsembleNet'):
            print("Loading Ensemble model (FGSM + PGD + Mixed)...")
            net = project_module.EnsembleNet()
        else:
            print("Error: EnsembleNet not found in model.py")
            sys.exit(1)
    else:
        net = project_module.Net()
        net.load_for_testing(project_dir=args.project_dir)
    
    net.to(device)
    net.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=transform)
    valid_loader = get_validation_loader(cifar, batch_size=args.batch_size)

    # Get randomization function from model.py if available
    randomization_fn = None
    if args.randomization:
        if hasattr(project_module, 'randomization_layer'):
            randomization_fn = project_module.randomization_layer
            print("Randomization defense ENABLED (using {} samples per image)".format(args.num_samples))
        else:
            print("Warning: randomization_layer not found in model.py, running without randomization")
    
    acc_nat = test_natural(net, valid_loader, num_samples=args.num_samples, 
                          use_randomization=args.randomization, randomization_fn=randomization_fn)
    print("Model nat accuracy (test): {:.2f}%".format(acc_nat))

if __name__ == "__main__":
    main()
