"""
EPU-CNN Model Package

This package contains the implementation of E Pluribus Unum CNN (EPU-CNN),
an interpretable CNN architecture for image classification.
"""

from model.epu import EPU, BaseEPU
from model.layers import (
    AdditiveLayer,
    ConvolutionalLayer2D,
    ConvSubnetAVGBlock,
    ContributionHead,
    InterpretationLayer,
)
from model.subnetworks import SubnetAVG, BaseSubnet

__all__ = [
    "EPU",
    "BaseEPU",
    "AdditiveLayer",
    "ConvolutionalLayer2D",
    "ConvSubnetAVGBlock",
    "ContributionHead",
    "InterpretationLayer",
    "SubnetAVG",
    "BaseSubnet",
]
