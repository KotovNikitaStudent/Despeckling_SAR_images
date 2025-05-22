import torch.nn as nn

from .despecklenet import DespeckleNet, Conv2dSame
from .despecklenetpluplus import DespeckleNetPlusPlus
from .cbamdilatednet import CBAMDilatedNet
from .multiscalereconstructionnet import MultiScaleReconstructionNet


def weights_init(m):
    """
    Инициализация весов для всех поддерживаемых типов слоёв.
    """
    if isinstance(m, Conv2dSame):
        nn.init.kaiming_normal_(
            m.conv.weight, mode="fan_out", nonlinearity="leaky_relu"
        )
        if m.conv.bias is not None:
            nn.init.constant_(m.conv.bias, 0)

    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, (nn.GroupNorm, nn.InstanceNorm2d)):
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.constant_(m.weight, 1)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, (nn.Linear,)):
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


__all__ = [
    "DespeckleNet",
    "DespeckleNetPlusPlus",
    "CBAMDilatedNet",
    "MultiScaleReconstructionNet",
    "weights_init",
]
