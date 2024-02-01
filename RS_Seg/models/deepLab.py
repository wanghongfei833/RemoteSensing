import numpy as np
import torch
from models.deeplab.network import modeling


def create_model(num_class, in_channels):
    model_name = 'deeplabv3plus_mobilenet'
    model = modeling.__dict__[model_name](num_classes=num_class, output_stride=8, inchannel=in_channels)
    return model


def deeplabel_v3p(num_class, in_channels):
    model_name = 'deeplabv3plus_mobilenet'
    model = modeling.__dict__[model_name](num_classes=num_class, output_stride=8, inchannel=in_channels)
    model.model_name = "deeplabv3p"
    return model
