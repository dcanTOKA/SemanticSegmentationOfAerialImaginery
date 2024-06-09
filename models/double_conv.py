import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    (Conv2d -> Batch -> ReLU) x2

    For CNN calculation :
        H_out = ((H_in + (2 * padding) - kernel_size ) / stride ) + 1
        W_out = ((W_in + (2 * padding) - kernel_size ) / stride ) + 1

    Example : input = (256, 256, 3) and in_channel 3
        -> nn.Conv2d(3, 64, kernel_size=3, padding=1)
            out = (256, 256, 64)
        -> nn.BatchNorm2d(64)
            out = (256, 256, 64)
        -> nn.ReLU(inplace=True)
            out = (256, 256, 64)
        -> nn.Conv2d(64, 64, kernel_size=3, padding=1)
            out = (256, 256, 64)
        -> nn.BatchNorm2d(out_channels)
            out = (256, 256, 64)
        -> nn.ReLU(inplace=True)
            out = (256, 256, 64)

    Returns :
        input   : (batch_size, in_channel, 256, 256)
        output  : (batch_size, out_channel, 256, 256)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()

        self.conv_pipe = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_pipe(x)
