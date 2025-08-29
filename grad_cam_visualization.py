import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from .utils import load_config, ensure_dir

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def forward(self, input_img):
        return self.model(input_img)
    
    def generate_cam(self, input_image, target_class=None):
        device = next(self.model.parameters()).device
        
        # Forward pass
        output = self.forward(input_image)
        
        if target_class is None:
            target_class = np.argmax(output.cpu().data.numpy())
        
        # Zero gradients
        self.model.zero_grad()
        
        # Target for backprop
        one_hot_output = torch.zeros_like(output)
        one_hot_output[0][target_class] = 1
        
        # Backward pass
        output.backward(gradient=one_hot_output, retain_graph=True)
        
        # Check if gradients and activations were captured
        if self.gradients is None or self.activations is None:
            raise ValueError("Gradients or activations not captured. Check hook registration.")
        
        # Get gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weight the activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # Apply ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize
        cam = cv2.resize(cam, input_image.shape[2:][::-1])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam) if np.max(cam) > 0 else cam
        
        return cam, target_class

def visualize_grad_cam(model, dataloader, class_names, num_images=5):
    config = load_config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Get the target layer - using a different approach
    target_layer = model.layer4[2].conv3
    
    # Initialize Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    # Get a batch of images
    try:
        data_iter = iter(dataloader['val'])
        images, labels = next(data_iter)
    except:
        data_iter = iter(dataloader['test'])
        images, labels = next(data_iter)
    
    # Select images
    indices = range(min(num_images, len(images)))
    
    # Create figure
    fig, axes = plt.subplots(nrows=len(indices), ncols=3, figsize=(15, 5*len(indices)))
    
    if len(indices) == 1:
        axes = axes.reshape(1, -1)
    
    successful_visualizations = 0
    
    for i, idx in enumerate(indices):
        try:
            image = images[idx].unsqueeze(0).to(device)
            label = labels[idx].item()
            
            # Get original image for visualization
            original_image = image.squeeze().cpu().numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            original_image = std * original_image + mean
            original_image = np.clip(original_image, 0, 1)
            
            # Generate CAM
            cam, predicted_class = grad_cam.generate_cam(image)
            
            # Apply colormap to CAM
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Superimpose heatmap on original image
            superimposed_img = 0.6 * np.float32(original_image) + 0.4 * heatmap
            superimposed_img = superimposed_img / np.max(superimposed_img)
            
            # Plot original image
            axes[i, 0].imshow(original_image)
            axes[i, 0].set_title(f'Original: {class_names[label]}')
            axes[i, 0].axis('off')
            
            # Plot heatmap
            axes[i, 1].imshow(cam, cmap='jet')
            axes[i, 1].set_title('Grad-CAM Heatmap')
            axes[i, 1].axis('off')
            
            # Plot superimposed image
            axes[i, 2].imshow(superimposed_img)
            axes[i, 2].set_title(f'Predicted: {class_names[predicted_class]}')
            axes[i, 2].axis('off')
            
            successful_visualizations += 1
            
        except Exception as e:
            print(f"Error processing image {i}: {e}")
            # Show error message
            for j in range(3):
                axes[i, j].text(0.5, 0.5, f'Error: {str(e)[:50]}...', 
                               ha='center', va='center', transform=axes[i, j].transAxes)
                axes[i, j].set_title('Error')
                axes[i, j].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    ensure_dir(config['paths']['results'] + '/grad_cam_visualizations')
    plt.savefig(config['paths']['results'] + '/grad_cam_visualizations/grad_cam_results.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Successfully generated {successful_visualizations} Grad-CAM visualizations!")
    
    return fig