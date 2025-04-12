import sys
sys.path.append("path/to/epu-cnn-torch")
import torch
import numpy as np

from torchvision.transforms.functional import InterpolationMode
from utils.epu_utils import (
    EPUDataset, trainer, EPUConfig, 
    TensorboardLoggerCallback, EarlyStoppingCallback
)
from utils.custom_transforms import ImageToPFM, PFMToTensor
from torch.utils.data import DataLoader
from torchvision import transforms
from model.epu import EPU
from glob import glob
import os


def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epu_config = EPUConfig.yaml_load("configs/binary_epu_config.yaml", key_config="epu")
    train_parameters = EPUConfig.yaml_load("configs/binary_epu_config.yaml", key_config="train_parameters")

    epu = EPU(epu_config)

    # This is an example on how to train the model on Banapple dataset: https://github.com/innoisys/Banapple
    train_data = glob("../data/banapple/train/*")
    train_labels = np.asarray([1 if "apple" in d.split("\\")[-1] else 0 for d in train_data], dtype=np.float32)
    
    validation_data = glob("../data/banapple/validation/*")
    validation_labels = np.asarray([1 if "apple" in d.split("\\")[-1] else 0 for d in validation_data], dtype=np.float32)

    dataset = EPUDataset(train_data, 
                         train_labels,
                         transforms= transforms.Compose([
                                     transforms.Resize((train_parameters.input_size, train_parameters.input_size), 
                                                       interpolation=InterpolationMode.BICUBIC),
                                     transforms.RandomHorizontalFlip(),
                                     ImageToPFM(train_parameters.input_size),
                                     PFMToTensor()]),
                                     cache_size=1666)
    
    validation_dataset = EPUDataset(validation_data, 
                         validation_labels,
                         transforms= transforms.Compose([
                                     transforms.Resize((train_parameters.input_size, train_parameters.input_size), 
                                                       interpolation=InterpolationMode.BICUBIC),
                                     ImageToPFM(train_parameters.input_size),
                                     PFMToTensor()]),
                                     cache_size=1000)

    train_loader = DataLoader(dataset, 
                            batch_size=train_parameters.batch_size, 
                            shuffle=train_parameters.shuffle, 
                            num_workers=train_parameters.num_workers, 
                            pin_memory=train_parameters.pin_memory,
                            persistent_workers=train_parameters.persistent_workers)

    validation_loader = DataLoader(validation_dataset, 
                            batch_size=train_parameters.batch_size, 
                            shuffle=train_parameters.shuffle, 
                            num_workers=train_parameters.num_workers, 
                            pin_memory=train_parameters.pin_memory,
                            persistent_workers=train_parameters.persistent_workers)

    # Set up output directories
    experiment_name = f"{epu_config.model_name}_{train_parameters.epochs}epochs"
    log_dir = os.path.join("logs", experiment_name)
    checkpoint_path = os.path.join("checkpoints", f"{experiment_name}.pt")
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # Initialize callbacks
    callbacks = [
        # TensorboardLogger to track metrics
        TensorboardLoggerCallback(
            log_dir=log_dir,
            log_histograms=True  # Set to False if you want to save space/computation
        ),
        # EarlyStopping to prevent overfitting
        EarlyStoppingCallback(
            patience=25,  # Stop if no improvement for 10 epochs
            delta=0.001,  # Minimum change to count as improvement
            checkpoint_path=checkpoint_path,
            verbose=True
        )
    ]

    print(f"Using device: {device}")
    print(f"Logging to: {log_dir}")
    print(f"Best model will be saved to: {checkpoint_path}")
    
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(epu.parameters(), lr=float(train_parameters.learning_rate))
    
    # Train with callbacks
    trained_model = trainer(
            model=epu, 
            criterion=criterion, 
            optimizer=optimizer, 
            train_loader=train_loader, 
            val_loader=validation_loader, 
            epochs=int(train_parameters.epochs), 
            device=device,
            callbacks=callbacks  # Pass the callbacks list here
    )
    
    # Save the final model
    final_model_path = os.path.join("models", f"{experiment_name}_final.pt")
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save(trained_model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}")

if __name__ == "__main__":
    main()


