import torch
import torch.nn as nn

from models.double_conv import DoubleConv
from models.down_sample import DownSample
from models.up_sample import UpSample


class UNet(nn.Module):
    """
    Example input : (256, 256, 3)

        DownSample(3, 64) -----------> (128, 128, 64)
        DownSample(64, 128) ---------> (64, 64, 128)
        DownSample(128, 256) --------> (32, 32, 256)
        DownSample(256, 512) --------> (16, 16, 512)

        DoubleConv(512, 1024) -------> (16, 16, 1024)

        UpSample(1024, 512) ---------> (32, 32, 512)
        UpSample(512, 256)  ---------> (64, 64, 256)
        UpSample(256, 128)  ---------> (128, 128, 128)
        UpSample(128, 64)   ---------> (256, 256, 64)

        nn.Conv2d(64, num_classes, kernel_size=1)
            out = 256, 256, num_classes

    Returns :
        input: (batch_size, in_channel, H, W)
        output: (batch_size, num_classes, H, W)

    """
    def __init__(self, in_channel, num_classes):
        super().__init__()

        self.down_1 = DownSample(in_channel, 64)
        self.down_2 = DownSample(64, 128)
        self.down_3 = DownSample(128, 256)
        self.down_4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(512, 1024)

        self.up_1 = UpSample(1024, 512)
        self.up_2 = UpSample(512, 256)
        self.up_3 = UpSample(256, 128)
        self.up_4 = UpSample(128, 64)

        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        down_1, pool_1 = self.down_1(x)
        down_2, pool_2 = self.down_2(pool_1)
        down_3, pool_3 = self.down_3(pool_2)
        down_4, pool_4 = self.down_4(pool_3)

        bottle_neck = self.bottle_neck(pool_4)

        up_1 = self.up_1(bottle_neck, down_4)
        up_2 = self.up_2(up_1, down_3)
        up_3 = self.up_3(up_2, down_2)
        up_4 = self.up_4(up_3, down_1)

        return self.out(up_4)


if __name__ == "__main__":
    double_conv = DoubleConv(256, 256)
    print(double_conv)

    input_image_dummy = torch.rand((1, 3, 512, 512))
    model = UNet(3, 5)
    out = model(input_image_dummy)

    print(out.shape)