from typing import List
from yaml import YAMLObject
from collections import namedtuple

import torch
import pickle
import cv2 as cv
import numpy as np
import seaborn as sns
import torch.nn as nn
import matplotlib.pyplot as plt

from numpy.typing import ArrayLike

from model.layers import AdditiveLayer
from utils.epu_utils import module_mapping, SubnetworkConfig


# No use at the moment will be utilized in the future
EPUClassificationResult = namedtuple("EPUClassificationResult", 
                                     ["class_label", 
                                      "contributions",
                                      "bias", 
                                      "interpretations"])

class BaseEPU(nn.Module):

    def __init__(self, 
                 n_subnetworks: int, 
                 subnetwork: str,
                 n_classes: int,
                 subnetwork_config: SubnetworkConfig,
                 epu_activation: str = "sigmoid",
                 categorical_input_features: List[str] = None):
        
        super(BaseEPU, self).__init__()
        

        self._n_subnetworks = n_subnetworks
        self._subnetwork = module_mapping(subnetwork)
        self._subnetworks = nn.ModuleList([self._subnetwork(subnetwork_config) 
                                           for _ in range(n_subnetworks)])
        self._additive_layer = AdditiveLayer(layer_activation=epu_activation, n_classes=n_classes)
        
        # No use at the moment will be utilized in the future
        self._classification_result = EPUClassificationResult(class_label=None, 
                                                              contributions=None, 
                                                              bias=None, 
                                                              interpretations=None)
        
        self._categorical_input_features = categorical_input_features
        self._interpretations = None
        self._input_features = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._input_features = x
        self._interpretations = [subnetwork(_x) for _x, subnetwork in zip(x, self._subnetworks)]
        return self._additive_layer(torch.stack(self._interpretations))

    def get_interpretations(self) -> ArrayLike:
        if self._interpretations is None:
            return None
        return [interpretation.detach().cpu().numpy() for interpretation in self._interpretations]
    
    def get_additive_layer_bias(self) -> ArrayLike:
        return self._additive_layer._bias.detach().cpu().numpy()
    
    def plot_rss(self, savefig: bool=False, fig_name: str="rss.png"):
        plt.xlim(-1, 1)
        data = {}
        for i, input_feature_name in enumerate(self._categorical_input_features):
            data[input_feature_name] = self._interpretations[i].detach().cpu().numpy()

        sns.barplot(x=[float(v) for arr in data.values() for v in arr.flatten()], y=list(data.keys()),
                    palette=['red' if x < 0 else 'green' for x in data.values()])
        plt.yticks(rotation=45)
        plt.show()
        if savefig:
            plt.savefig(fig_name)
        plt.close()

    def get_prm(self, block_idx: int=3, 
                savefig: bool=False, 
                refine_prm: bool=False, 
                *args, 
                **kwargs) -> ArrayLike:
        
        prms = {}
        for (subnetwork, input_feature_name) in zip(self._subnetworks, self._categorical_input_features):
            prms[input_feature_name] = subnetwork.get_prm(block_idx)
        
        input_image = kwargs.get("input_image", np.zeros((kwargs.get("height", 256), kwargs.get("width", 256), 3)).astype(np.uint8))
        width, height = input_image.size

        if refine_prm:
            for key, value in prms.items():
                prms[key] = subnetwork.refine_prm(value, 
                                                  height, 
                                                  width)
        else:
            for key, value in prms.items():
                prms[key] = cv.resize(value, (width, height), interpolation=cv.INTER_CUBIC)

        
        if savefig:
            n = len(prms)  # number of keys/overlays

            # Create a figure with n rows and 2 columns
            fig, axes = plt.subplots(n, 2, figsize=(10, 5 * n))

            # If there is only one key, axes won't be 2D so we wrap it in a list
            if n == 1:
                axes = [axes]

            # Loop through prms and plot each pair of subplots
            for ax, (key, value) in zip(axes, prms.items()):
                # Left subplot: the original image
                ax[0].imshow(input_image)
                ax[0].set_title('Original Image')
                ax[0].axis('off')
                
                # Right subplot: original image with overlay
                ax[1].imshow(input_image)
                ax[1].imshow(value, alpha=0.5, cmap='jet')
                ax[1].set_title(f'{key} PRM Overlay')
                ax[1].axis('off')

            plt.tight_layout()
            plt.savefig("interpretations/all_prm.png")
            plt.show()
        return prms


class EPU(BaseEPU):
    
    def __init__(self, config: YAMLObject):
        
        super(EPU, self).__init__(n_subnetworks=config.n_subnetworks, 
                                  subnetwork=config.subnetwork, 
                                  n_classes=config.n_classes,
                                  epu_activation=config.epu_activation,
                                  subnetwork_config=config.subnetwork_architecture,
                                  categorical_input_features=config.categorical_input_features)
    
    @staticmethod
    def load_model(config_path: str, weights_path: str):
        try:
            with open(config_path, "rb") as f:
                config = pickle.load(f)
        except Exception as e:
            raise ValueError(f"Error loading config file: {e}")
        
        model = EPU(config)
        model.load_state_dict(torch.load(weights_path))
        return model