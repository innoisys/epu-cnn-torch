import sys
import os

# Add the project root directory to the Python path
# This assumes scripts/eval.py is in the scripts directory under the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import torch
import numpy as np
import argparse
import json

from torchvision.transforms.functional import InterpolationMode
from utils.epu_utils import EPUDataset, validate, EPUConfig
from utils.custom_transforms import ImageToPFM, PFMToTensor
from torch.utils.data import DataLoader
from torchvision import transforms
from model.epu import EPU
from glob import glob
from datetime import datetime


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate EPU model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model')
    parser.add_argument('--config_path', type=str, default="configs/binary_epu_config.yaml", 
                        help='Path to the model configuration file')
    parser.add_argument('--test_data', type=str, required=True, 
                        help='Path to test data directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='eval_results', 
                        help='Directory to save evaluation results')
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load configurations
    epu_config = EPUConfig.yaml_load(args.config_path, key_config="epu")
    train_parameters = EPUConfig.yaml_load(args.config_path, key_config="train_parameters")
    
    # Initialize the model
    model = EPU(epu_config)
    
    # Load model weights
    print(f"Loading model from {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    
    # Load test data
    print(f"Loading test data from {args.test_data}")
    test_data = glob(os.path.join(args.test_data, "*"))
    
    # Generate labels (assuming same naming convention as in train.py)
    test_labels = np.asarray([1 if "apple" in d.split("/")[-1] else 0 
                           for d in test_data], dtype=np.float32)
    
    # Create dataset
    test_dataset = EPUDataset(test_data, 
                            test_labels,
                            transforms=transforms.Compose([
                                transforms.Resize((train_parameters.input_size, train_parameters.input_size), 
                                                  interpolation=InterpolationMode.BICUBIC),
                                ImageToPFM(train_parameters.input_size),
                                PFMToTensor()]),
                            cache_size=1000)
    
    # Create data loader
    test_loader = DataLoader(test_dataset, 
                           batch_size=args.batch_size, 
                           shuffle=False,  # No need to shuffle for evaluation
                           num_workers=train_parameters.num_workers, 
                           pin_memory=train_parameters.pin_memory,
                           persistent_workers=train_parameters.persistent_workers)
    
    # Define loss function
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Evaluate model
    print("Starting evaluation...")
    metrics, loss, (predictions, targets) = validate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        desc="Evaluating",
        return_predictions=True
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(args.output_dir, f"eval_results_{timestamp}.json")
    
    # Prepare results dictionary
    results = {
        "model_path": args.model_path,
        "config_path": args.config_path,
        "test_data": args.test_data,
        "metrics": metrics,
        "loss": loss,
        "num_samples": len(test_data),
        "timestamp": timestamp
    }
    
    # Save results to JSON file
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Evaluation complete. Results saved to {results_file}")
    
    # Optional: compute confusion matrix and additional metrics for binary classification
    if epu_config.n_classes == 1:
        from sklearn.metrics import confusion_matrix, classification_report
        
        # Convert predictions to binary labels
        binary_preds = (predictions > 0.5).astype(int).ravel()
        binary_targets = targets.ravel()
        
        # Compute confusion matrix
        cm = confusion_matrix(binary_targets, binary_preds)
        
        # Generate classification report
        report = classification_report(binary_targets, binary_preds, output_dict=True)
        
        # Save additional metrics
        confusion_file = os.path.join(args.output_dir, f"confusion_matrix_{timestamp}.txt")
        report_file = os.path.join(args.output_dir, f"classification_report_{timestamp}.json")
        
        # Save confusion matrix
        with open(confusion_file, 'w') as f:
            f.write("Confusion Matrix:\n")
            f.write(f"TN: {cm[0][0]}, FP: {cm[0][1]}\n")
            f.write(f"FN: {cm[1][0]}, TP: {cm[1][1]}\n")
        
        # Save classification report
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"Additional metrics saved to {args.output_dir}")
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        print(f"TN: {cm[0][0]}, FP: {cm[0][1]}")
        print(f"FN: {cm[1][0]}, TP: {cm[1][1]}")


if __name__ == "__main__":
    main() 