import os
import yaml
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import models

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

def create_model():
    """Create and return the ResNet-50 model with custom classifier"""
    config = load_config()
    model_name = config['model']['name']
    pretrained = config['model']['pretrained']
    num_classes = config['model']['num_classes']
    dropout_rate = config['model']['dropout_rate']
    
    # Load pre-trained ResNet-50
    if model_name == "resnet50":
        # Use the new weights parameter instead of pretrained
        if pretrained:
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            model = models.resnet50(weights=None)
        
        # Freeze early layers
        for param in model.parameters():
            param.requires_grad = False
        
        # Replace the final fully connected layer
        num_ftrs = model.fc.in_features
        
        # Add dropout and a new FC layer
        model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_ftrs, num_classes)
        )
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    return model