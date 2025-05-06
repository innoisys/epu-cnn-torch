import os
import sys
import argparse

from glob import glob
from pathlib import Path

import torch
import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.transforms.functional import InterpolationMode

from model.epu import EPU
from utils.epu_utils import (
    trainer, EPUConfig, 
    TensorboardLoggerCallback, EarlyStoppingCallback,
    launch_tensorboard, module_mapping
)
from utils.data_utils import (
    FilenameDatasetParser, FolderDatasetParser, 
    EPUDataset
)
from utils.mappings import custom_module_mapping
from utils.custom_transforms import ImageToPFM, PFMToTensor


epu_path = Path(__file__).resolve().parent
sys.path.append(str(epu_path))


def data_prep(train_parameters: EPUConfig):

    train_data = custom_module_mapping(train_parameters.dataset_parser)(dataset_path=train_parameters.dataset_path, 
                                       mode="train", 
                                       label_mapping=train_parameters.label_mapping,
                                       image_extension=train_parameters.image_extension)
    
    validation_data = custom_module_mapping(train_parameters.dataset_parser)(dataset_path=train_parameters.dataset_path, 
                                            mode="validation", 
                                            label_mapping=train_parameters.label_mapping,
                                            image_extension=train_parameters.image_extension)

    dataset = EPUDataset(train_data,
                         transforms= transforms.Compose([
                                     transforms.Resize((train_parameters.input_size, train_parameters.input_size), 
                                                       interpolation=InterpolationMode.BICUBIC),
                                     transforms.RandomHorizontalFlip(),
                                     ImageToPFM(train_parameters.input_size),
                                     PFMToTensor()]),
                                     cache_size=500)
    
    validation_dataset = EPUDataset(validation_data, 
                         transforms= transforms.Compose([
                                     transforms.Resize((train_parameters.input_size, train_parameters.input_size), 
                                                       interpolation=InterpolationMode.BICUBIC),
                                     ImageToPFM(train_parameters.input_size),
                                     PFMToTensor()]),
                                     cache_size=250)

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

    return train_loader, validation_loader


def user_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--tensorboard", required=False, action="store_true", help="Launches tensorboard on port 6006")
    args = parser.parse_args()
    return args


def main():
    
    args = user_arguments()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epu_config = EPUConfig.yaml_load(args.config_path, key_config="epu")
    train_parameters = EPUConfig.yaml_load(args.config_path, key_config="train_parameters")

    # This is an example on how to train the model on Banapple dataset: https://github.com/innoisys/Banapple
    train_loader, validation_loader = data_prep(train_parameters)

    # Set up output directories
    default_id = 0
    while os.path.exists(os.path.join("logs", f"{epu_config.model_name}_{train_parameters.epochs}epochs_{default_id}")):
        default_id += 1
    
    experiment_name = f"{epu_config.model_name}_{train_parameters.epochs}epochs_{default_id}"
    
    epu_config.set_attribute("experiment_name", experiment_name)
    epu_config.set_attribute("label_mapping", train_parameters.label_mapping.__dict__)
    epu_config.set_attribute("confidence", 0.5)
    epu_config.set_attribute("mode", train_parameters.mode)
    train_parameters.set_attribute("experiment_name", experiment_name)

    log_dir = os.path.join("logs", experiment_name)
    checkpoint_path = os.path.join(f"checkpoints/{experiment_name}", f"{experiment_name}.pt")
    config_path = os.path.join(f"checkpoints/{experiment_name}", f"epu.config")
    train_config_path = os.path.join(f"checkpoints/{experiment_name}", f"train.config")

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    epu_config.save_config_object(config_path)
    train_parameters.save_config_object(train_config_path)

    epu = EPU(epu_config)

    launch_tensorboard(launch=args.tensorboard)
    
    # Initialize callbacks
    callbacks = [
        # TensorboardLogger to track metrics
        TensorboardLoggerCallback(
            log_dir=log_dir,
            log_histograms=True  # Set to False if you want to save space/computation
        ),
        # EarlyStopping to prevent overfitting
        EarlyStoppingCallback(
            patience=train_parameters.early_stopping_patience,  # Stop if no improvement for 10 epochs
            delta=0.001,  # Minimum change to count as improvement
            checkpoint_path=checkpoint_path,
            verbose=True
        )
    ]

    print(f"🚀 Using device: {device}")
    print(f"📝 Logging to: {log_dir}")
    print(f"💾 Best model will be saved to: {checkpoint_path}")
    
    criterion = module_mapping(train_parameters.loss)()
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
            callbacks=callbacks,
            mode=train_parameters.mode,
            n_classes=len(train_parameters.label_mapping.__dict__))
    
    # Save the final model
    final_model_path = os.path.join(f"checkpoints/{experiment_name}", f"{experiment_name}_final.pt")
    torch.save(trained_model.state_dict(), final_model_path)
    
    print(f"💾 Final model saved to: {final_model_path}")


if __name__ == "__main__":
    main()
