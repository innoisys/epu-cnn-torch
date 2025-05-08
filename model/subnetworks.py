import torch
import cv2 as cv
import numpy as np
import torch.nn as nn

from copy import deepcopy
from typing import Dict, Optional
from numpy.typing import ArrayLike
from skimage.filters import threshold_yen
from skimage.measure import shannon_entropy
from model.layers import ConvSubnetAVGBlock, ContributionHead
from utils.epu_utils import (module_mapping, SubnetworkConfig, 
                             min_max_normalization)


class BaseSubnet(object):
    
    def __init__(self):
        # Variables to track intermediate activations
        self._feature_maps = {}
        self._last_input = None
        self._last_output = None

    def get_feature_maps(self) -> Dict[str, torch.Tensor]:
        """Get all intermediate feature maps."""
        return self._feature_maps

    def get_last_input(self) -> Optional[torch.Tensor]:
        """Get the last input tensor."""
        return self._last_input

    def get_last_output(self) -> Optional[torch.Tensor]:
        """Get the last output tensor."""
        return self._last_output

    def refine_prm(self, prm: ArrayLike, height: int, width: int) -> ArrayLike:
        temp_prm = cv.resize(prm, (width, height), interpolation=cv.INTER_CUBIC)
        thresh = threshold_yen(prm)
        temp_prm = (temp_prm > thresh) * temp_prm
        temp_prm = cv.applyColorMap(temp_prm, cv.COLORMAP_JET)
        return temp_prm

    def get_prm(self, block_idx: int= 3) -> ArrayLike:

        feature_maps = self._blocks[block_idx - 1].get_feature_maps().detach().cpu().numpy()
        batch, channels, height, width = feature_maps.shape
        entropies = []

        if batch > 1:
            raise ValueError("Batch size must be 1, no implementation yet for batch size > 1")

        for k in range(channels):
            normalized_feature_map = min_max_normalization(feature_maps[0, k, :, :])
            channel_entropy = shannon_entropy(normalized_feature_map)
            entropies.append(channel_entropy)

        sorted_entropies = deepcopy(entropies)
        sorted_entropies.sort(reverse=True)
        sorted_entropies = sorted_entropies[:len(sorted_entropies) // 2]

        selected_feature_maps = [min_max_normalization(feature_maps[0, entropies.index(sorted_entropy), :, :]) * 255 for sorted_entropy in sorted_entropies]
        prm = np.asarray(selected_feature_maps).mean(axis=0).astype(np.uint8)

        return prm


class SubnetAVG(BaseSubnet, nn.Module):

    def __init__(self, 
                 block_config: SubnetworkConfig):
        
        nn.Module.__init__(self)
        BaseSubnet.__init__(self)

        self._blocks = nn.ModuleList([ConvSubnetAVGBlock(getattr(block_config, 
                                                                 f"block_{i + 1}")) for i in range(block_config.n_blocks)])

        if block_config.has_pooling:
            self._pooling = module_mapping(block_config.pooling_type)((1, 1))
        else:
            self._pooling = None
        
        if block_config.has_contribution_head:
            self._contribution_head = ContributionHead(in_features=block_config.contribution_head.in_features, 
                                                        n_classes=block_config.n_classes,
                                                        n_hidden_layers=block_config.contribution_head.n_hidden_layers,
                                                        n_hidden_neurons=block_config.contribution_head.n_hidden_neurons,
                                                        hidden_activation=block_config.contribution_head.hidden_activation,
                                                        output_activation=block_config.contribution_head.output_activation)
        else:
            self._contribution_head = None
        
        self._n_classes = block_config.n_classes
        
        # Track intermediate activations
        self._feature_maps = {}
        self._last_input = None
        self._last_output = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._last_input = x
        self._feature_maps['input'] = x
        
        # Process through blocks
        for i, block in enumerate(self._blocks):
            x = block(x)
            self._feature_maps[f'block_{i}'] = x
            
        if self._pooling is not None:
            x = self._pooling(x)
            self._feature_maps['pooling'] = x
            
        x = torch.flatten(x, 1)
        self._feature_maps['flattened'] = x
        
        if self._contribution_head is not None:
            x = self._contribution_head(x)
            self._feature_maps['contribution'] = x
            
        self._last_output = x
        self._feature_maps['output'] = x
        
        return x
