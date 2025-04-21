import os
import pickle

from glob import glob
from typing import (List, Union, 
                    Callable, Dict, 
                    Tuple, Optional)
from yaml import safe_load
from collections import OrderedDict
from typing import (List, Callable, 
                    Dict, Tuple, 
                    Optional, Any)

import torch
import cv2 as cv
import numpy as np
import torch.nn as nn

from PIL import Image
from tqdm import tqdm

from numpy.typing import ArrayLike
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import InterpolationMode
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

from utils.custom_transforms import ImageToPFM, PFMToTensor


class TensorboardLogger(object):
    """
    Example Usage:
        logger = TensorboardLogger('logs/my_experiment')

        model = torchvision.models.resnet18()
        input_sample = torch.rand(1, 3, 224, 224)

        # Log model graph
        logger.log_model_graph(model, input_sample)

        # Dummy training loop
        for epoch in range(10):
            dummy_loss = torch.rand(1).item()

            # Log scalar
            logger.log_scalar('Loss/train', dummy_loss, epoch)

            # Log histograms
            logger.log_histogram(model, epoch)

        # Dummy images
        images = torch.rand(16, 3, 64, 64)
        logger.log_images('sample_images', images, step=0)

        logger.close()

    """

    def __init__(self, log_dir='runs/experiment'):
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, scalar_dict: Dict[str, float], step: int):
        for tag, value in scalar_dict.items():
            self.writer.add_scalar(tag, value, step)

    def log_histogram(self, model: nn.Module, step: int):
        for name, param in model.named_parameters():
            self.writer.add_histogram(f'weights/{name}', param, step)
            if param.grad is not None:
                self.writer.add_histogram(f'gradients/{name}', param.grad, step)

    def log_images(self, tag: str, images: torch.Tensor, step: int, nrow: int = 4):
        img_grid = torchvision.utils.make_grid(images, nrow=nrow)
        self.writer.add_image(tag, img_grid, step)

    def log_model_graph(self, model: nn.Module, input_sample: torch.Tensor):
        self.writer.add_graph(model, input_sample)

    def close(self):
        self.writer.close()


class EarlyStopping(object):
    def __init__(self, patience: int = 10, delta: float = 0, checkpoint_path: str = 'checkpoint.pt', verbose: bool = True):
        """
        Example Usage:
            model = torch.nn.Linear(10, 1)
            arly_stopping = EarlyStopping(patience=3, verbose=True)

            for epoch in range(100):
                # Training and validation steps here
                val_loss = torch.rand(1).item()  # Example validation loss

                print(f'Epoch {epoch}, Validation Loss: {val_loss:.4f}')
                early_stopping(val_loss, model)

                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    break

            # Load best model
            model.load_state_dict(torch.load('checkpoint.pt'))

        Args:
            patience (int): How many epochs to wait after last improvement before stopping.
            delta (float): Minimum change to qualify as an improvement.
            checkpoint_path (str): File path to save the best model.
            verbose (bool): If True, prints messages when improvement occurs.
        """

        self.patience = patience
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss: float, model: nn.Module):
        score = -val_loss  # Because lower loss is better

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            print(f"New best score: {score}")
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: nn.Module):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model...')
        torch.save(model.state_dict(), self.checkpoint_path)
        self.val_loss_min = val_loss


# Dummy implementation of LRUCache, it will be removed
class LRUCache(object):
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if key not in self.cache:
            self.misses += 1
            return None
        self.hits += 1
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            # Update existing value and move to end
            self.cache.move_to_end(key)
            self.cache[key] = value
        else:
            if len(self.cache) >= self.capacity:
                # Remove least recently used item
                self.cache.popitem(last=False)
            self.cache[key] = value

    def clear(self):
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self):
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self.cache)
        }


class BlockConfig(object):
    """Configuration for a network layer block."""
    def __init__(self, **entries):
        for key, _ in entries.items():
            setattr(self, key, entries.get(key, None))


class SubnetworkConfig(object):
    """Configuration for the subnetwork architecture."""
    def __init__(self, **entries):
        for block_name, block_config in entries.items():
            if isinstance(block_config, dict) and block_name != 'classification_head':
                # Create attribute directly instead of using a dictionary
                setattr(self, block_name, BlockConfig(**block_config))
            elif isinstance(block_config, dict) and block_name == 'classification_head':
                setattr(self, block_name, ClassificationHeadConfig(**block_config))
            else:
                setattr(self, block_name, block_config)


class ClassificationHeadConfig(object):
    """Configuration for the classification head."""
    def __init__(self, **entries):
        for key, _ in entries.items():
            setattr(self, key, entries.get(key, None))


class EPUConfig(object):
    """Configuration for the EPU model.
    
    Example structure from YAML:
    epu:
        n_subnetworks: 4
        subnetwork: "subnetavg"
        subnetwork_architecture:
            block_one:
                in_channels: 1
                out_channels: 64
                ...
        n_classes: 1
        epu_activation: "sigmoid"
        subnetwork_activation: "tanh"
        categorical_input_features: [...]
    """
    def __init__(self, **entries):
        for key, value in entries.items():
            if key == 'ep':
                setattr(self, key, SubnetworkConfig(**value))
            elif isinstance(value, dict):
                setattr(self, key, EPUConfig(**value))
            else:
                setattr(self, key, value)

    @staticmethod
    def yaml_load(file_path: str, key_config: str="epu") -> 'EPUConfig':
        """Load configuration from a YAML file.
        
        Args:
            file_path: Path to the YAML configuration file
            key_config: Top-level key in the YAML file (default: "epu")
            
        Returns:
            EPUConfig object with nested SubnetworkConfig and LayerConfig objects
        """
        with open(file_path, "r") as f:
            config = safe_load(f)
        return EPUConfig(**config[key_config])

    def __repr__(self):
        """Pretty print the configuration."""
        attrs = []
        for key, value in self.__dict__.items():
            attrs.append(f"{key}={value}")
        return f"EPUConfig({', '.join(attrs)})"

    def save_config_object(self, path: str):
        """Save the configuration to a file."""
        try:
            with open(path, "wb") as f:
                pickle.dump(self, f)
        except Exception as e:
            print(f"Error saving config object: {e}")
    
    def set_attribute(self, key: str, value: Any):
        setattr(self, key, value)

    @staticmethod
    def load_config_object(path: str):
        """Load the configuration from a file."""
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading config object: {e}")


class FolderDatasetParser(object):
    def __init__(self, 
                 dataset_path: str, 
                 mode: str = "train",
                 label_mapping: Union[Dict[str, int], EPUConfig] = {"apple": 1, "banana": 0},
                 image_extension: str = "jpg"):
        
        if isinstance(label_mapping, dict):
            self._label_mapping = label_mapping
        elif isinstance(label_mapping, EPUConfig):
            self._label_mapping = label_mapping.__dict__

        self._mode = mode
        self._image_extension = image_extension
        self._dataset_path = dataset_path
        self._dataset_folders = glob(f"{self._dataset_path}/{self._mode}/*")
        self._filenames = []
        self._labels = []
        self._parse_dataset_folders()
        self._sanity_check()

    def _sanity_check(self):
        if len(self._filenames) == 0:
            raise ValueError(f"No files found in {self._dataset_path}")
        if len(self._labels) == 0:
            raise ValueError(f"No labels found in {self._dataset_path}")

    def _parse_dataset_folders(self):
        for folder in self._dataset_folders:
            filenames = glob(f"{folder}/*.{self._image_extension}")
            n_filenames = len(filenames)
            self._filenames += filenames
            label = self._label_mapping.get(os.path.basename(folder), None)
            if label is None:
                raise ValueError(f"No label for data in {folder}")
            self._labels += np.repeat(label, n_filenames).tolist()
        self._labels = np.array(self._labels, dtype=np.float32)
    
    @property
    def filenames(self) -> List[str]:
        return self._filenames

    @property
    def labels(self) -> ArrayLike:
        return self._labels


class FilenameDatasetParser(object):
    
    def __init__(self, 
                 dataset_path: str, 
                 mode: str = "train", 
                 label_mapping: Union[Dict[str, int], EPUConfig] = {"apple": 1, "banana": 0},
                 image_extension: str = "jpg"):
        
        if isinstance(label_mapping, dict):
            self._label_mapping = label_mapping
        elif isinstance(label_mapping, EPUConfig):
            self._label_mapping = label_mapping.__dict__

        self._dataset_path = f"{dataset_path}/{mode}"
        self._filenames = glob(f"{self._dataset_path}/*.{image_extension}")
        self._labels = self._get_labels()
        self._sanity_check()

    def _sanity_check(self):
        if len(self._filenames) == 0:
            raise ValueError(f"No files found in {self._dataset_path}")
        if len(self._labels) == 0:
            raise ValueError(f"No labels found in {self._dataset_path}")
        assert len(self._filenames) == len(self._labels), "Number of files and labels do not match"

    def _get_labels(self) -> ArrayLike:
        labels = []
        for filename in self._filenames:
            filename = os.path.basename(filename)
            miss = False
            for key, value in self._label_mapping.items():
                if key in filename:
                    labels.append(value)
                    miss = True
                    break
            if not miss:
                raise ValueError(f"Label not found in {filename}")
        return np.array(labels, dtype=np.float32)

    @property
    def filenames(self) -> List[str]:
        return self._filenames

    @property
    def labels(self) -> ArrayLike:
        return self._labels


class EPUDataset(Dataset):

    def __init__(self, data: FilenameDatasetParser, transforms: Callable = None, cache_size: int = 1000):
        self._data = data.filenames
        self._labels = data.labels
        self._transforms = transforms
        self._cache = LRUCache(capacity=cache_size)
        self._preload_images()

    def _preload_images(self):
        """Preload images into memory for faster access"""
        for idx in tqdm(range(len(self._data)), desc="Preloading images"):
            if idx < self._cache.capacity:
                image = Image.open(self._data[idx])
                self._cache.put(idx, image)

    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx):
        # Try to get from cache first
        image = self._cache.get(idx)
        if image is None:
            # If not in cache, load and transform
            image = Image.open(self._data[idx])
            self._cache.put(idx, image)
        
        if self._transforms is not None:
                image = self._transforms(image)
        
        return {
            "image": image,
            "label": self._labels[idx]
        }

    def clear_cache(self):
        """Clear the image cache to free memory"""
        self._cache.clear()

    def get_cache_stats(self):
        """Get cache statistics"""
        return self._cache.get_stats()


class TensorboardLoggerCallback(object):
    """Adapter to use TensorboardLogger as a callback in the trainer function."""
    
    def __init__(self, log_dir='runs/experiment', log_histograms=True):
        self.logger = TensorboardLogger(log_dir=log_dir)
        self.log_histograms = log_histograms
    
    def on_training_begin(self, state):
        # Optionally log model graph if we had an input sample
        pass
    
    def on_epoch_end(self, state):
        epoch = state['epoch']
        
        # Log losses
        self.logger.log_scalar('Loss/train', state['train_loss'], epoch)
        self.logger.log_scalar('Loss/val', state['val_loss'], epoch)
        
        # Log training metrics
        train_metrics = state['train_metrics']
        for metric, value in train_metrics.items():
            self.logger.log_scalar(f'Metrics/train/{metric}', value, epoch)
        
        # Log validation metrics
        val_metrics = state['val_metrics']
        for metric, value in val_metrics.items():
            self.logger.log_scalar(f'Metrics/val/{metric}', value, epoch)
        
        # Log model weights and gradients
        if self.log_histograms:
            self.logger.log_histogram(state['model'], epoch)
    
    def on_training_end(self, state):
        self.logger.close()


class EarlyStoppingCallback(object):
    """Adapter to use EarlyStopping as a callback in the trainer function."""
    
    def __init__(self, patience=10, delta=0, checkpoint_path='checkpoint.pt', verbose=True):
        self.early_stopping = EarlyStopping(
            patience=patience,
            delta=delta,
            checkpoint_path=checkpoint_path,
            verbose=verbose
        )
        self.checkpoint_path = checkpoint_path
    
    def on_training_begin(self, state):
        # Set the best model path in the state
        state['best_model_path'] = self.checkpoint_path
    
    def on_validation_end(self, state):
        # Call early stopping with validation loss
        self.early_stopping(state['val_loss'], state['model'])
        
        # Update early stop flag in state
        state['early_stop'] = self.early_stopping.early_stop


def preprocess_image(image: Union[str, Image.Image, np.typing.ArrayLike], input_size: int) -> torch.Tensor:
    
    """Process a single image for inference."""
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size), 
                            interpolation=InterpolationMode.BICUBIC),
        ImageToPFM(input_size),
        PFMToTensor()
    ])

    if isinstance(image, str):
        image = Image.open(image)
        image = transform(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
        mode = image.mode
        if mode != "RGB":
            image = image.convert("RGB")
        image = transform(image)
    elif isinstance(image, Image.Image):
        image = transform(image)
    else:
        raise ValueError(f"Invalid image type: {type(image)}")        

    return image


def preprocess_images(images: List[Union[str, Image.Image, np.typing.ArrayLike]], 
                      input_size: int) -> torch.Tensor:
    return torch.stack([preprocess_image(image, input_size) for image in images])


def trainer(model: nn.Module, 
            criterion: nn.Module, 
            optimizer: torch.optim.Optimizer, 
            train_loader: DataLoader, 
            val_loader: DataLoader, 
            epochs: int, 
            device: torch.device, 
            callbacks: List = None) -> nn.Module:
    """Train a model using the provided data loaders and optimization parameters.

    Args:
        model: The neural network model to train
        criterion: Loss function to optimize
        optimizer: Optimization algorithm
        train_loader: DataLoader containing training data
        val_loader: DataLoader containing validation data 
        epochs: Number of training epochs
        device: Device to run training on (cuda/cpu)
        callbacks: List of callback objects to use during training (e.g., EarlyStopping, TensorboardLogger)

    Returns:
        The trained model
    """
    # Initialize callbacks list if not provided
    callbacks = callbacks or []
    
    # Dictionary to store training state that will be passed to callbacks
    state = {
        'model': model,
        'early_stop': False,
        'best_model_path': None,
    }
    
    # Call on_training_begin for each callback
    for callback in callbacks:
        if hasattr(callback, 'on_training_begin'):
            callback.on_training_begin(state)
   
    model.to(device)
    for epoch in tqdm(range(epochs), desc="Training"):
        state['epoch'] = epoch
        
        # Call on_epoch_begin for each callback
        for callback in callbacks:
            if hasattr(callback, 'on_epoch_begin'):
                callback.on_epoch_begin(state)
        
        model.train()
        train_loss = 0
        train_predictions = []
        train_targets = []
        
        for i, sample in enumerate(tqdm(train_loader, desc="Training Sample Loop")):
            x, y = sample["image"], sample["label"]
            x, y = torch.stack(x).to(device), y.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y.float())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_predictions.append(y_hat.detach().cpu().numpy())
            train_targets.append(y.cpu().numpy())
            
            # Call on_batch_end for each callback
            batch_state = {
                'batch': i,
                'loss': loss.item(),
                'size': len(train_loader)
            }
            for callback in callbacks:
                if hasattr(callback, 'on_batch_end'):
                    callback.on_batch_end({**state, **batch_state})
        
        # Calculate training metrics
        train_predictions = np.vstack(train_predictions)
        train_targets = np.vstack(train_targets)
        train_metrics = calculate_metrics(train_targets, train_predictions, train_predictions)
        train_loss = train_loss / len(train_loader)
        
        # Update state with training results
        state['train_loss'] = train_loss
        state['train_metrics'] = train_metrics
        
        # Call on_training_end for each callback
        for callback in callbacks:
            if hasattr(callback, 'on_training_end'):
                callback.on_training_end(state)
        
        # Validate using the standalone validation function
        val_metrics, val_loss, _ = validate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
            desc="Validating"
        )
        
        # Update state with validation results
        state['val_loss'] = val_loss
        state['val_metrics'] = val_metrics
        
        # Call on_validation_end for each callback
        for callback in callbacks:
            if hasattr(callback, 'on_validation_end'):
                callback.on_validation_end(state)
        
        # Print epoch results
        print(f"\n[+]Epoch: {epoch + 1}")
        print(f"[+]Train Loss: {train_loss:.4f}")
        print(f"[+]Train Metrics - Acc: {train_metrics['accuracy']:.4f}, AUC: {train_metrics['auc']:.4f}, "
              f"Precision: {train_metrics['precision']:.4f}, Recall: {train_metrics['recall']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}")
        print(f"[+]Val Loss: {val_loss:.4f}")
        print(f"[+]Val Metrics - Acc: {val_metrics['accuracy']:.4f}, AUC: {val_metrics['auc']:.4f}, "
              f"Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}")
        
        # Get cache statistics from the dataset
        if hasattr(train_loader.dataset, 'get_cache_stats'):
            cache_stats = train_loader.dataset.get_cache_stats()
            print(f"[+]Cache Stats - Hit Rate: {cache_stats['hit_rate']:.2%}, "
                  f"Size: {cache_stats['size']}/{train_loader.dataset._cache.capacity}")
        
        # Call on_epoch_end for each callback
        for callback in callbacks:
            if hasattr(callback, 'on_epoch_end'):
                callback.on_epoch_end(state)
        
        # Check for early stopping
        if state.get('early_stop', False):
            print("Early stopping triggered")
            if state.get('best_model_path'):
                print(f"Loading best model from {state['best_model_path']}")
                model.load_state_dict(torch.load(state['best_model_path']))
            break
    
    # Call on_training_end for each callback
    for callback in callbacks:
        if hasattr(callback, 'on_training_end'):
            callback.on_training_end(state)
    
    return model


def min_max_normalization(x: np.ndarray) -> np.ndarray:
    """Normalize the input array to the range [0, 1]."""
    return (x - x.min()) / (x.max() - x.min() + 1e-6)


def standardization(x: np.ndarray) -> np.ndarray:
    """Standardize the input array to have zero mean and unit variance."""
    return (x - x.mean()) / (x.std() + 1e-6)


def validate(model: nn.Module,
            data_loader: DataLoader,
            criterion: nn.Module,
            device: torch.device,
            desc: str = "Validating",
            return_predictions: bool = False) -> Tuple[Dict[str, float], float, Optional[Tuple[np.ndarray, np.ndarray]]]:
    """Validate/test a model using the provided data loader.
    
    Args:
        model: The neural network model to validate
        data_loader: DataLoader containing validation/test data
        criterion: Loss function to compute
        device: Device to run validation on (cuda/cpu)
        desc: Description for the progress bar
        return_predictions: Whether to return predictions and targets
        
    Returns:
        Tuple containing:
        - Dictionary of metrics (accuracy, precision, recall, f1, auc)
        - Average loss value
        - (Optional) Tuple of (predictions, targets) if return_predictions is True
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for sample in tqdm(data_loader, desc=desc):
            x, y = sample["image"], sample["label"]
            x, y = torch.stack(x).to(device), y.to(device).unsqueeze(1)
            y_hat = model(x)
            loss = criterion(y_hat, y.float())
            
            total_loss += loss.item()
            all_predictions.append(y_hat.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    # Stack all predictions and targets
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)
    
    # Calculate metrics
    metrics = calculate_metrics(all_targets, all_predictions, all_predictions)
    avg_loss = total_loss / len(data_loader)
    
    # Print metrics
    print(f"\n[+]Evaluation Results:")
    print(f"[+]Loss: {avg_loss:.4f}")
    print(f"[+]Metrics - Acc: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}, "
          f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, "
          f"F1: {metrics['f1']:.4f}")
    
    if return_predictions:
        return metrics, avg_loss, (all_predictions, all_targets)
    return metrics, avg_loss, None


def load_model(model_path: str, config_path: str):
    from model.epu import EPU
    """Load a trained EPU model."""
    # Load configuration
    epu_config = EPUConfig.load_config_object(config_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model and load weights
    model = EPU(epu_config)
    model.to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model


def module_mapping(module: str) -> nn.Module:
    from model.layers import AdditiveLayer, ConvSubnetAVGBlock
    from model.subnetworks import SubnetAVG
    
    modules = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "softmax": nn.Softmax,
        "leakyrelu": nn.LeakyReLU,
        "prelu": nn.PReLU,
        "elu": nn.ELU,
        "selu": nn.SELU,
        "gelu": nn.GELU,
        "swish": nn.SiLU,
        "mish": nn.Mish,
        "silu": nn.SiLU,
        "mish": nn.Mish,
        "linear": nn.Identity,
        "logsigmoid": nn.LogSigmoid,
        "softplus": nn.Softplus,
        "softsign": nn.Softsign,
        "tanhshrink": nn.Tanhshrink,
        "hardshrink": nn.Hardshrink,
        "softshrink": nn.Softshrink,
        "tanhshrink": nn.Tanhshrink,
        "hardtanh": nn.Hardtanh,
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "globalaveragepooling": nn.AdaptiveAvgPool2d,
        "batchnorm1d": nn.BatchNorm1d,
        "batchnorm2d": nn.BatchNorm2d,
        "batchnorm3d": nn.BatchNorm3d,
        "instancenorm1d": nn.InstanceNorm1d,
        "instancenorm2d": nn.InstanceNorm2d,
        "instancenorm3d": nn.InstanceNorm3d,
        "layer_norm": nn.LayerNorm,
        "group_norm": nn.GroupNorm,
        "local_response_norm": nn.LocalResponseNorm,
        "dropout": nn.Dropout,
        "alpha_dropout": nn.AlphaDropout,
        "feature_alpha_dropout": nn.FeatureAlphaDropout,
        "dropout2d": nn.Dropout2d,
        "dropout3d": nn.Dropout3d,
        "conv_subnet_avg_block": ConvSubnetAVGBlock,
        "additive_layer": AdditiveLayer,
        "subnetavg": SubnetAVG,
        "maxpooling3d": nn.MaxPool3d,
        "maxpooling2d": nn.MaxPool2d,
        "maxpooling1d": nn.MaxPool1d,
    }

    try:
        return modules[module.lower()]
    except KeyError as e:
        available_modules = list(modules.keys())
        raise ValueError(f"Module {module} not found in the module mapping. Available modules are: {available_modules}") from e


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, float]:
    """Calculate various classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels (after threshold for binary, or argmax for multiclass)
        y_prob: Predicted probabilities (for AUC calculation)
        
    Returns:
        Dictionary containing the calculated metrics
    """
    metrics = {}
    
    # Handle binary and multiclass cases
    average_method = 'binary' if y_true.shape[1] == 1 else 'macro'
    
    # Convert predictions to appropriate format
    if y_true.shape[1] == 1:  # Binary case
        y_true = y_true.ravel()
        y_pred = (y_pred > 0.5).astype(int).ravel()
        if y_prob is not None:
            y_prob = y_prob.ravel()
    else:  # Multiclass case
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        
    # Calculate metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average=average_method, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=average_method, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average=average_method, zero_division=0)
    
    # Calculate AUC if probabilities are provided
    if y_prob is not None:
        try:
            if average_method == "binary":  # Binary case
                metrics['auc'] = roc_auc_score(y_true, y_prob)
            else:  # Multiclass case
                metrics['auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
        except ValueError:
            metrics['auc'] = float('nan')  # Handle cases where AUC cannot be calculated
            
    return metrics