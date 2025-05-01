import sys
import os

from glob import glob
from pathlib import Path
from datetime import datetime

epu_path = Path(__file__).resolve().parent
sys.path.append(str(epu_path))

import torch
import numpy as np
import argparse
import json

from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader
from torchvision import transforms

from model.epu import EPU
from utils.data_utils import EPUDataset
from utils.epu_utils import validate, EPUConfig, load_model, module_mapping
from utils.custom_transforms import ImageToPFM, PFMToTensor
from utils.mappings import custom_module_mapping


def user_arguments() -> argparse.Namespace:
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate EPU model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the directory containing the saved model')
    parser.add_argument('--test_data', type=str, required=True, 
                        help='Path to test data directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='eval_results', 
                        help='Directory to save evaluation results')
    parser.add_argument('--confidence', type=float, default=0.5, help='Classifier confidence threshold')
    args = parser.parse_args()
    return args


def data_prep(dataset_path, train_parameters: EPUConfig = None) -> DataLoader:
    dataset_parser = custom_module_mapping(train_parameters.dataset_parser)(dataset_path=dataset_path, 
                                mode="test", 
                                label_mapping=train_parameters.label_mapping,
                                image_extension=train_parameters.image_extension)
    
    dataset = EPUDataset(dataset_parser,
                         transforms= transforms.Compose([
                                     transforms.Resize((train_parameters.input_size, train_parameters.input_size), 
                                                       interpolation=InterpolationMode.BICUBIC),
                                     transforms.RandomHorizontalFlip(),
                                     ImageToPFM(train_parameters.input_size),
                                     PFMToTensor()]),
                                     cache_size=1666)
    
       # Create data loader
    dataset = DataLoader(dataset, 
                           batch_size=train_parameters.batch_size, 
                           shuffle=False,  # No need to shuffle for evaluation
                           num_workers=train_parameters.num_workers, 
                           pin_memory=train_parameters.pin_memory,
                           persistent_workers=train_parameters.persistent_workers)

    return dataset


def main():

    args = user_arguments()
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load configurations
    # format the path additing the cwd in an appropriate format for different operating systems
    epu_config_path = os.path.join(os.getcwd(), *args.model_path.split("/"), "epu.config")
    train_config_path = os.path.join(os.getcwd(), *args.model_path.split("/"), "train.config")

    train_parameters = EPUConfig.load_config_object(train_config_path)
    epu_config = EPUConfig.load_config_object(epu_config_path)

    # Load model weights
    print(f"Loading model from {args.model_path}")
    checkpoint_path = os.path.join(os.getcwd(), *args.model_path.split("/"), f"{epu_config.experiment_name}.pt")
    model = load_model(checkpoint_path, epu_config_path, 
                       mode=train_parameters.mode, 
                       label_mapping=train_parameters.label_mapping.__dict__, 
                       confidence=args.confidence)
    
    # Load test data
    print(f"Loading test data from {args.test_data}")
    test_loader = data_prep(args.test_data, train_parameters)
    
    # Define loss function
    criterion = module_mapping(train_parameters.loss)()
    
    # Evaluate model
    print("Starting evaluation...")
    metrics, loss, (predictions, targets) = validate(
        model=model,
        criterion=criterion,
        device=device,
        desc="Evaluating",
        return_predictions=True,
        mode=train_parameters.mode,
        data_loader=test_loader,
        n_classes=epu_config.n_classes
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.join(os.getcwd(), args.output_dir, epu_config.experiment_name), exist_ok=True)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(os.getcwd(), args.output_dir, epu_config.experiment_name, f"eval_results_{timestamp}.json")
    
    # Prepare results dictionary
    results = {
        "model_path": checkpoint_path,
        "config_path": epu_config_path,
        "train_config_path": train_config_path,
        "test_data": args.test_data,
        "metrics": metrics,
        "loss": loss,
        "num_samples": len(test_loader),
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
        binary_preds = (predictions > args.confidence).astype(int).ravel()
        binary_targets = targets.ravel()
        
        # Compute confusion matrix
        cm = confusion_matrix(binary_targets, binary_preds)
        
        # Generate classification report
        report = classification_report(binary_targets, binary_preds, output_dict=True)
        
        # Save additional metrics
        eval_results_dir = os.path.join(os.getcwd(), args.output_dir, epu_config.experiment_name)
        confusion_file = os.path.join(eval_results_dir, f"confusion_matrix_{timestamp}.txt")
        report_file = os.path.join(eval_results_dir, f"classification_report_{timestamp}.json")
        
        # Save confusion matrix
        with open(confusion_file, 'w') as f:
            f.write("Confusion Matrix:\n")
            f.write(f"TN: {cm[0][0]}, FP: {cm[0][1]}\n")
            f.write(f"FN: {cm[1][0]}, TP: {cm[1][1]}\n")
        
        # Save classification report
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"Additional metrics saved to {eval_results_dir}")
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        print(f"TN: {cm[0][0]}, FP: {cm[0][1]}")
        print(f"FN: {cm[1][0]}, TP: {cm[1][1]}")
    
    # Multiclass classification metrics
    elif epu_config.n_classes > 1:
        from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
        import numpy as np
        
        # Convert predictions to class labels
        class_preds = np.argmax(predictions, axis=1)
        class_targets = np.argmax(targets, axis=1)
        
        # Compute confusion matrix
        cm = confusion_matrix(class_targets, class_preds)
        
        # Generate classification report
        report = classification_report(class_targets, class_preds, output_dict=True)
        
        # Compute per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(class_targets, class_preds)
        
        # Save additional metrics
        eval_results_dir = os.path.join(os.getcwd(), args.output_dir, epu_config.experiment_name)
        confusion_file = os.path.join(eval_results_dir, f"confusion_matrix_{timestamp}.txt")
        report_file = os.path.join(eval_results_dir, f"classification_report_{timestamp}.json")
        per_class_file = os.path.join(eval_results_dir, f"per_class_metrics_{timestamp}.json")
        
        # Save confusion matrix
        with open(confusion_file, 'w') as f:
            f.write("Confusion Matrix:\n\n")
            # Write column headers
            f.write("Predicted →\n")
            f.write("Actual ↓\n\n")
            # Write the matrix with aligned columns
            max_width = len(str(np.max(cm)))
            format_str = f'%{max_width}d'
            for i in range(cm.shape[0]):
                row = [format_str % x for x in cm[i]]
                f.write('  '.join(row) + '\n')
        
        # Save classification report
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=4)
        
        # Save per-class metrics
        per_class_metrics = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'support': support.tolist()
        }
        with open(per_class_file, 'w') as f:
            json.dump(per_class_metrics, f, indent=4)
        
        print(f"Additional metrics saved to {eval_results_dir}")
        
        # Print summary metrics
        print("\nClassification Report:")
        print(classification_report(class_targets, class_preds))
        
        print("\nPer-class Metrics:")
        for i in range(epu_config.n_classes):
            print(f"Class {i}:")
            print(f"  Precision: {precision[i]:.4f}")
            print(f"  Recall: {recall[i]:.4f}")
            print(f"  F1-score: {f1[i]:.4f}")
            print(f"  Support: {support[i]}")


if __name__ == "__main__":
    main() 