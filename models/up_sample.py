import torch.nn as nn
import torch

from models.double_conv import DoubleConv


class UpSample(nn.Module):
    """
        (nn.ConvTranspose2d(in_channel, in_channel // 2, kernel_size=2, stride=2) -> DoubleConv) x1

    For ConvTranspose calculation :
        H_out = (H_in -1) * stride - (2 * padding) + kernel_size + padding
        W_out = (W_in -1) * stride - (2 * padding) + kernel_size + padding

    Example : input = (28, 28, 1024)
        -> nn.ConvTranspose2d(1024, 1024 // 2, kernel_size=2, stride=2)
            out = (56, 56, 1024)
        -> DoubleConv(1024, 512)
            out = (56, 56, 512)

    Returns :
        input   : (batch_size, in_channel, H, W)
        output  : (batch_size, out_channel, H*2, W*2)
    """
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channel, in_channel // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channel, out_channel)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)
