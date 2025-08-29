import argparse
import numpy as np
import torch
from src.model_training import run_training
from src.evaluation import evaluate_model, plot_confusion_matrix, plot_training_history, generate_classification_report
from src.grad_cam_visualization import visualize_grad_cam
from src.data_preprocessing import create_data_loaders
from src.utils import load_config, load_model, create_model  # Fixed import

def main():
    parser = argparse.ArgumentParser(description='Pneumonia Detection with ResNet-50 and Grad-CAM')
    parser.add_argument('--mode', type=str, default='all', 
                        choices=['train', 'evaluate', 'visualize', 'all'],
                        help='Mode to run: train, evaluate, visualize, or all')
    args = parser.parse_args()
    
    config = load_config()
    
    if args.mode in ['train', 'all']:
        print("Training the model...")
        model, history, class_names = run_training()
        
        # Plot training history
        plot_training_history(history)
    
    if args.mode in ['evaluate', 'all']:
        print("Evaluating the model...")
        
        # Load model
        model = create_model()
        model_path = config['paths']['trained_models'] + '/pneumonia_resnet50.pth'
        model = load_model(model, model_path)
        
        # Create data loaders
        dataloaders, dataset_sizes, class_names = create_data_loaders()
        
        # Evaluate on test set
        y_pred, y_true, y_probs = evaluate_model(model, dataloaders['test'], class_names)
        
        # Generate metrics and plots
        cm = plot_confusion_matrix(y_true, y_pred, class_names)
        report = generate_classification_report(y_true, y_pred, class_names)
        
        print(f"Test Accuracy: {sum(np.array(y_pred) == np.array(y_true)) / len(y_true):.4f}")
    
    if args.mode in ['visualize', 'all']:
        print("Generating Grad-CAM visualizations...")
        
        # Load model
        model = create_model()
        model_path = config['paths']['trained_models'] + '/pneumonia_resnet50.pth'
        model = load_model(model, model_path)
        # Create data loaders
        dataloaders, dataset_sizes, class_names = create_data_loaders()
        
        # Generate Grad-CAM visualizations
        visualize_grad_cam(model, dataloaders, class_names, num_images=5)

if __name__ == '__main__':
    main()