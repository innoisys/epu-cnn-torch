
from glob import glob
from collections import OrderedDict
from typing import Dict, Union, List, Callable
from abc import ABC, abstractmethod

import os
import numpy as np

from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from numpy.typing import ArrayLike
from torch.utils.data import Dataset, DataLoader

from utils.epu_utils import EPUConfig


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


class DatasetPraser(ABC):

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
        self._filenames = []
        self._labels = []
        self._parse_dataset()
        self._sanity_check()
    
    def _sanity_check(self):
        if len(self._filenames) == 0:
            raise ValueError(f"No files found in {self._dataset_path}")
        if len(self._labels) == 0:
            raise ValueError(f"No labels found in {self._dataset_path}")
        assert len(self._filenames) == len(self._labels), "Number of files and labels do not match"
    
    @abstractmethod
    def _parse_dataset(self):
        """Parse the dataset
        This method is implemented by subclasses. It should parse the filenames and lables.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def filenames(self) -> List[str]:
        return self._filenames

    @property
    def labels(self) -> ArrayLike:
        return self._labels


class FolderDatasetParser(DatasetPraser):
    def __init__(self, 
                 dataset_path: str, 
                 mode: str = "train",
                 label_mapping: Union[Dict[str, int], EPUConfig] = {"apple": 1, "banana": 0},
                 image_extension: str = "jpg"):
        
        super(FolderDatasetParser, self).__init__(dataset_path, mode, label_mapping, image_extension)

    def _parse_dataset(self):
        _dataset_folders = glob(f"{self._dataset_path}/{self._mode}/*")
        for folder in _dataset_folders:
            filenames = glob(f"{folder}/*.{self._image_extension}")
            n_filenames = len(filenames)
            self._filenames += filenames
            label = self._get_label(folder)
            self._labels += np.repeat(label, n_filenames).tolist()
        self._labels = np.array(self._labels, dtype=np.float32)
    
    def _get_label(self, folder: str) -> int:
        label = self._label_mapping.get(os.path.basename(folder), None)
        if label is None:
            raise ValueError(f"No label for data in {folder}")
        return label


class FilenameDatasetParser(DatasetPraser):
    
    def __init__(self, 
                 dataset_path: str, 
                 mode: str = "train", 
                 label_mapping: Union[Dict[str, int], EPUConfig] = {"apple": 1, "banana": 0},
                 image_extension: str = "jpg"):
        
        super(FilenameDatasetParser, self).__init__(dataset_path, mode, label_mapping, image_extension)
    
    def _parse_dataset(self):
        self._filenames = glob(f"{self._dataset_path}/{self._mode}/*.{self._image_extension}")
        self._get_labels()

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


class HuggingFaceDatasetParser(DatasetPraser):
    """
    Parser for Hugging Face image datasets to be used with EPUDataset.

    Args:
        dataset_name (str): name of the dataset on Hugging Face hub, e.g. 'blanchon/FireRisk'.
        split (str): which split to load, e.g. 'train', 'test', etc.
        image_column (str): name of the image column feature (defaults to 'image').
        label_column (str): name of the label column feature (defaults to 'label').
        label_mapping (dict or EPUConfig, optional): maps label names to integers.
            If None, will infer from dataset's ClassLabel feature.
    """
    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        image_column: str = "image",
        label_column: str = "label",
        label_mapping: Union[Dict[str, int], EPUConfig, None] = None
    ):
        # Load the dataset
        self.dataset_name = dataset_name
        self.split = split
        ds = load_dataset(dataset_name)[split]

        # Determine label mapping
        if isinstance(label_mapping, dict):
            self._label_mapping = label_mapping
        elif isinstance(label_mapping, EPUConfig):
            self._label_mapping = label_mapping.__dict__
        else:
            # Infer from ClassLabel feature
            feature = ds.features[label_column]
            if hasattr(feature, 'names'):
                self._label_mapping = {name: i for i, name in enumerate(feature.names)}
            else:
                raise ValueError(
                    "Cannot infer label_mapping: provided label_column is not ClassLabel"
                )

        # Store for parsing
        self._image_column = image_column
        self._label_column = label_column
        self._filenames = []
        self._labels = []

        # Parse and sanity check
        self._parse_dataset(ds)
        self._sanity_check()

    def _parse_dataset(self, ds):
        for example in tqdm(ds, desc=f"Parsing {self.dataset_name}/{self.split}"):
            img = example[self._image_column]

            # Handle features.Image (PIL) and filepaths
            if isinstance(img, Image.Image) and hasattr(img, 'filename'):
                path = img.filename
            elif isinstance(img, str) and os.path.exists(img):
                path = img
            else:
                raise ValueError(
                    f"Unexpected image format for example: {type(img)}"
                )

            self._filenames.append(path)

            # Resolve label
            raw_lbl = example[self._label_column]
            if isinstance(raw_lbl, int):
                # Convert int to name then map
                feat = ds.features[self._label_column]
                lbl_name = feat.int2str(raw_lbl)
            else:
                lbl_name = str(raw_lbl)

            mapped = self._label_mapping.get(lbl_name)
            if mapped is None:
                raise ValueError(f"Label '{lbl_name}' not in mapping")
            self._labels.append(mapped)

        self._labels = np.array(self._labels, dtype=np.float32)

    @property
    def filenames(self):
        return self._filenames

    @property
    def labels(self):
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