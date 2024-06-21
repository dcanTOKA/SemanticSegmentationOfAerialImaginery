import torch.nn as nn

from models.double_conv import DoubleConv


class DownSample(nn.Module):
    """

    ###################3####
    #### Encoder Block ####
    #######################


    (DoubleConv -> 2D Max Pooling) x1

    For Max Pooling calculation :
        H_out = ((H_in - pool_size) / stride ) + 1
        W_out = ((W_in - pool_size) / stride ) + 1

    Example : input = (256, 256, 3)
        -> DoubleConv(3, 64)
            out = (256, 256, 64)
        -> nn.MaxPool2d(kernel_size=2, stride=2)
            out = (128, 128, 64)


    Returns :
        input   : (batch_size, in_channel, H, W)
        output  : (batch_size, out_channel, H/2, W/2)
    """
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv = DoubleConv(in_channel, out_channel)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        pool = self.max_pool(down)

        return down, pool
