from .deepLab import deeplabel_v3p
from .unet import unet
from .swin_transform import swin_res_unet


def create_model(num_classes, in_channels, model_name,loss_fun):
    assert model_name in ["swin", "unet", "deep"], ValueError("模型只能是swin | unet | deep")
    if model_name == "swin":
        return swin_res_unet(num_classes, in_channels,loss_fun)
    elif model_name == "unet":
        return unet(num_classes, in_channels,loss_fun)
    else:
        return deeplabel_v3p(num_classes, in_channels,loss_fun)

