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
                 categorical_input_features: List[str] = None,
                 experiment_name: str = None,
                 mode: str = "binary",
                 label_mapping: dict = None,
                 confidence: float = 0.5):
        
        super(BaseEPU, self).__init__()
        
        self._confidence = confidence
        self._label_mapping = label_mapping
        self._reverse_label_mapping = {v: k for k, v in label_mapping.items()}
        self._n_subnetworks = n_subnetworks
        self._subnetwork = module_mapping(subnetwork)
        self._subnetworks = nn.ModuleList([self._subnetwork(subnetwork_config) 
                                           for _ in range(n_subnetworks)])
        self._additive_layer = AdditiveLayer(layer_activation=epu_activation, n_classes=n_classes)
        self._experiment_name = None
        # No use at the moment will be utilized in the future
        self._classification_result = EPUClassificationResult(class_label=None, 
                                                              contributions=None, 
                                                              bias=None, 
                                                              interpretations=None)
        self._mode = mode
        self._experiment_name = experiment_name
        self._categorical_input_features = categorical_input_features

        self._output = None
        self._interpretations = None
        self._input_features = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._input_features = x
        self._interpretations = [subnetwork(_x) for _x, subnetwork in zip(x, self._subnetworks)]
        self._output = self._additive_layer(torch.stack(self._interpretations))
        return self._output

    def get_interpretations(self) -> ArrayLike:
        if self._interpretations is None:
            return None
        return [interpretation.detach().cpu().numpy() for interpretation in self._interpretations]
    
    def get_additive_layer_bias(self) -> ArrayLike:
        return self._additive_layer._bias.detach().cpu().numpy()
    
    def _get_rss_binary(self) -> ArrayLike:
        data = {}
        for i, input_feature_name in enumerate(self._categorical_input_features):
            data[input_feature_name] = self._interpretations[i].squeeze().detach().cpu().numpy()
        return data, self._reverse_label_mapping[1], self._reverse_label_mapping[0]

    def _get_rss_multiclass(self) -> ArrayLike:
        data = {}
        prediction = self._output.detach().cpu().numpy()
        predicted_class_idx = np.argmax(prediction).item()
        for i, input_feature_name in enumerate(self._categorical_input_features):
            data[input_feature_name] = self._interpretations[i].squeeze()[predicted_class_idx].detach().cpu().numpy()
        return data, self._reverse_label_mapping[predicted_class_idx], "Other"

    def get_rss(self) -> ArrayLike:
        return self._get_rss_binary() if self._mode == "binary" else self._get_rss_multiclass()

    def plot_rss(self, savefig: bool=False, *args, **kwargs):
        # Create figure and axis objects explicitly
        fig, ax = plt.subplots()
        
        plt.xlim(-1, 1)
        data, predicted_label, other_label = self.get_rss()
        
        # Create a custom colormap from red to green
        custom_cmap = plt.cm.RdYlGn  # Red-Yellow-Green colormap

        # Create the bar plot
        sns.barplot(x=[float(v) for arr in data.values() for v in arr.flatten()], 
                   y=list(data.keys()),
                   palette=['red' if x < 0 else 'green' for x in data.values()],
                   ax=ax)
        
        # Create a ScalarMappable for the colorbar
        norm = plt.Normalize(-1, 1)
        sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
        sm.set_array([])
        
        # Add colorbar with explicit axes reference
        cbar = plt.colorbar(sm, ax=ax)
        
        # Add custom ticks and labels to colorbar
        cbar.set_ticks([-1, 1])
        cbar.set_ticklabels([f"{other_label}", f"{predicted_label}"])
        
        plt.yticks(rotation=45)
        if savefig:
            import os
            os.makedirs(f"interpretations/{self._experiment_name}", exist_ok=True)
            plt.savefig(f"interpretations/{self._experiment_name}/{kwargs.get('input_image_name', 'input_image')}_rss.png")
        plt.show()
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
            import os
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
            os.makedirs(f"interpretations/{self._experiment_name}", exist_ok=True)
            plt.savefig(f"interpretations/{self._experiment_name}/{kwargs.get('input_image_name', 'input_image')}_all_prm.png")
            plt.show()
        return prms

    def set_experiment_name(self, experiment_name: str):
        self._experiment_name = experiment_name
    
    @property
    def experiment_name(self) -> str:
        return self._experiment_name
    
    @property
    def confidence(self) -> float:
        return self._confidence


class EPU(BaseEPU):
    
    def __init__(self, config: YAMLObject):
        
        super(EPU, self).__init__(n_subnetworks=config.n_subnetworks, 
                                  subnetwork=config.subnetwork, 
                                  n_classes=config.n_classes,
                                  epu_activation=config.epu_activation,
                                  subnetwork_config=config.subnetwork_architecture,
                                  categorical_input_features=config.categorical_input_features,
                                  experiment_name=config.experiment_name,
                                  mode=config.mode,
                                  label_mapping=config.label_mapping,
                                  confidence=config.confidence)
    
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