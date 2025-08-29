import os  # Added missing import
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .utils import load_config, ensure_dir

def get_data_transforms():
    config = load_config()
    image_size = config['data']['image_size']
    
    # Data augmentation and normalization for training
    # Just normalization for validation and testing
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms

def create_data_loaders():
    config = load_config()
    data_dir = config['data']['data_dir']
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']
    
    data_transforms = get_data_transforms()
    
    # Create datasets
    image_datasets = {
        x: datasets.ImageFolder(
            os.path.join(data_dir, x),
            data_transforms[x]
        )
        for x in ['train', 'val', 'test']
    }
    
    # Create dataloaders
    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=True if x == 'train' else False,
            num_workers=num_workers
        )
        for x in ['train', 'val', 'test']
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes
    
    return dataloaders, dataset_sizes, class_names