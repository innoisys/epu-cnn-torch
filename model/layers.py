import torch 
import torch.nn as nn

from typing import List, Tuple, Dict, Optional
from utils.epu_utils import module_mapping, LayerConfig

class AdditiveLayer(nn.Module):
    
    def __init__(self, 
                 layer_activation: nn.Module, 
                 n_classes: int):
        
        super(AdditiveLayer, self).__init__()
        self._subnetwork_activation = module_mapping(layer_activation)()
        self._n_classes = n_classes
        self._bias = nn.Parameter(torch.randn(n_classes), requires_grad=True)
        self._last_input = None
        self._last_output = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._last_input = x
        x = torch.sum(x, dim=0) + self._bias
        self._last_output = x
        x = self._subnetwork_activation(x)
        return x

    def get_last_input(self) -> Optional[torch.Tensor]:
        return self._last_input

    def get_last_output(self) -> Optional[torch.Tensor]:
        return self._last_output

class ConvSubnetAVGBlock(nn.Module):

    def __init__(self, 
                 layer_config: LayerConfig):
        
        super(ConvSubnetAVGBlock, self).__init__() 
        self.conv_layers = nn.ModuleList()
        self.layer_config = layer_config
        
        temp_in_channels = layer_config.in_channels
        for _ in range(layer_config.n_conv_layers):
            self.conv_layers.append(nn.Conv2d(temp_in_channels, 
                                                    layer_config.out_channels, 
                                                    kernel_size=layer_config.kernel_size, 
                                                    stride=layer_config.stride, 
                                                    padding=layer_config.padding))
            temp_in_channels = layer_config.out_channels
        
        self.activation = module_mapping(layer_config.activation)()
        if layer_config.has_norm:
            self.batch_normalization = module_mapping(layer_config.norm_type)(num_features=layer_config.out_channels)

        if layer_config.has_pooling:
            self.pooling = module_mapping(layer_config.pooling_type)(kernel_size=layer_config.pooling_kernel_size, stride=layer_config.pooling_stride)

        # Track intermediate activations
        self._feature_maps = None
        self._last_input = None
        self._last_output = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._last_input = x

        # Track conv layer outputs
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            x = self.activation(x)
            
        if self.layer_config.has_pooling:
            x = self.pooling(x)
            
        if self.layer_config.has_norm:
            x = self.batch_normalization(x)
        
        self._feature_maps = x
        return x

    def get_feature_maps(self) -> torch.Tensor:
        return self._feature_maps


class InterpretationLayer(object):

    def __init__(self, n_subnetworks: int):
        self._n_subnetworks = n_subnetworks
        self._interpretations = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._interpretations = x
        return x
