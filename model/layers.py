import torch 
import torch.nn as nn

from typing import Tuple, Optional, Union, List
from utils.epu_utils import module_mapping, BlockConfig


class AdditiveLayer(nn.Module):
    
    def __init__(self, 
                 layer_activation: "str", 
                 n_classes: int):
        
        super(AdditiveLayer, self).__init__()
        self._epu_activation = module_mapping(layer_activation)()
        self._n_classes = n_classes
        self._bias = nn.Parameter(torch.randn(n_classes), requires_grad=True)
        self._last_input = None
        self._last_output = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._last_input = x
        x = torch.sum(x, dim=0) + self._bias
        self._last_output = x
        x = self._epu_activation(x)
        return x

    def get_last_input(self) -> Optional[torch.Tensor]:
        return self._last_input

    def get_last_output(self) -> Optional[torch.Tensor]:
        return self._last_output


class ConvolutionalLayer2D(nn.Module):

    def __init__(self, in_channels: int, 
                 out_channels: int, 
                 kernel_size: Tuple[int, int], 
                 stride: Tuple[int, int], 
                 padding: int,
                 normalization: Optional["str"] = None,
                 activation: Optional["str"] = "linear"):
        
        super(ConvolutionalLayer2D, self).__init__()
        self._conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self._batch_norm = nn.BatchNorm2d(out_channels)
        self._activation = module_mapping(activation)()

        self._normalization = module_mapping(normalization)(out_channels) if normalization else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv(x)
        x = self._activation(x)
        if self._normalization:
            x = self._normalization(x)
        return x


class ConvSubnetAVGBlock(nn.Module):

    def __init__(self, 
                 block_config: BlockConfig):
        
        super(ConvSubnetAVGBlock, self).__init__() 
        self._conv_layers = nn.ModuleList()
        self._block_config = block_config
        
        temp_in_channels = block_config.in_channels
        for _ in range(block_config.n_conv_layers):
            self._conv_layers.append(ConvolutionalLayer2D(in_channels=temp_in_channels, 
                                                         out_channels=block_config.out_channels, 
                                                         kernel_size=block_config.kernel_size, 
                                                         stride=block_config.stride, 
                                                         padding=block_config.padding,
                                                         normalization=block_config.norm_type,
                                                         activation=block_config.activation))
            temp_in_channels = block_config.out_channels
        
        if block_config.has_pooling:
            self._pooling = module_mapping(block_config.pooling_type)(kernel_size=block_config.pooling_kernel_size, stride=block_config.pooling_stride)

        # Track intermediate activations
        self._feature_maps = None
        self._last_input = None
        self._last_output = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._last_input = x

        # Track conv layer outputs
        for conv_layer in self._conv_layers:
            x = conv_layer(x)
            
        if self._block_config.has_pooling:
            x = self._pooling(x)
        
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


class ContributionHead(nn.Module):

    def __init__(self, 
                 in_features: int,
                 n_classes: int,
                 n_hidden_layers: Optional[int] = None,
                 n_hidden_neurons: Optional[Union[int, Tuple[int, ...]]] = None,
                 hidden_activation: Optional["str"] = "relu",
                 output_activation: "str" = "tanh"):
        
        super(ContributionHead, self).__init__()
        self._hidden_layers = nn.ModuleList()
        self._n_hidden_layers = n_hidden_layers if isinstance(n_hidden_layers, int) else 0
        self._n_hidden_neurons = n_hidden_neurons if isinstance(n_hidden_neurons, int) else None
        self._n_classes = n_classes
        self._hidden_activation = module_mapping(hidden_activation)()
        self._output_activation = module_mapping(output_activation)()
        self._fc = nn.Linear(in_features, n_classes)

        for _ in range(self._n_hidden_layers):
            self._hidden_layers.append(nn.Linear(in_features, self._n_hidden_neurons))
            in_features = self._n_hidden_neurons

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for hidden_layer in self._hidden_layers:
            x = hidden_layer(x)
            x = self._hidden_activation(x)
        x = self._fc(x)
        x = self._output_activation(x)
        return x

