from torch import nn
import torch

from models.conv import Conv
from models.double_conv import DoubleConv
from models.down_sample import DownSample

import torch.nn.functional as F


class Unet3Plus(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, deep_sup = True):
        super().__init__()

        self.filters = [64, 128, 256, 512, 1024]
        self.deep_sup = deep_sup
        self.num_classes = num_classes

        # Encoder

        self.e1 = DownSample(in_channels, self.filters[0])
        self.e2 = DownSample(self.filters[0], self.filters[1])
        self.e3 = DownSample(self.filters[1], self.filters[2])
        self.e4 = DownSample(self.filters[2], self.filters[3])

        # Bottleneck

        self.bottleneck = DoubleConv(self.filters[3], self.filters[4])

        # CGM: Classification Guided Module

        self.cgm = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(self.filters[-1], 2, kernel_size=1, padding=0),
            nn.AdaptiveMaxPool2d(1),
            nn.Sigmoid()
        )

        # Decoder 4

        self.e1_d4 = Conv(self.filters[0], self.filters[0])
        self.e2_d4 = Conv(self.filters[1], self.filters[0])
        self.e3_d4 = Conv(self.filters[2], self.filters[0])
        self.e4_d4 = Conv(self.filters[3], self.filters[0])
        self.e5_d4 = Conv(self.filters[4], self.filters[0])

        self.d4 = Conv(self.filters[0] * 5, self.filters[0])

        # Decoder 3

        self.e1_d3 = Conv(self.filters[0], self.filters[0])
        self.e2_d3 = Conv(self.filters[1], self.filters[0])
        self.e3_d3 = Conv(self.filters[2], self.filters[0])
        self.e4_d3 = Conv(self.filters[0], self.filters[0])
        self.e5_d3 = Conv(self.filters[4], self.filters[0])

        self.d3 = Conv(self.filters[0] * 5, self.filters[0])

        # Decoder 2

        self.e1_d2 = Conv(self.filters[0], self.filters[0])
        self.e2_d2 = Conv(self.filters[1], self.filters[0])
        self.e3_d2 = Conv(self.filters[0], self.filters[0])
        self.e4_d2 = Conv(self.filters[0], self.filters[0])
        self.e5_d2 = Conv(self.filters[4], self.filters[0])

        self.d2 = Conv(self.filters[0] * 5, self.filters[0])

        # Decoder 1

        self.e1_d1 = Conv(self.filters[0], self.filters[0])
        self.e2_d1 = Conv(self.filters[0], self.filters[0])
        self.e3_d1 = Conv(self.filters[0], self.filters[0])
        self.e4_d1 = Conv(self.filters[0], self.filters[0])
        self.e5_d1 = Conv(self.filters[4], self.filters[0])

        self.d1 = Conv(self.filters[0] * 5, self.filters[0])

        # Deep Supervision

        if self.deep_sup:
            self.y1 = nn.Conv2d(in_channels=self.filters[0], out_channels=self.num_classes, kernel_size=3, padding=1)
            self.y2 = nn.Conv2d(in_channels=self.filters[0], out_channels=self.num_classes, kernel_size=3, padding=1)
            self.y3 = nn.Conv2d(in_channels=self.filters[0], out_channels=self.num_classes, kernel_size=3, padding=1)
            self.y4 = nn.Conv2d(in_channels=self.filters[0], out_channels=self.num_classes, kernel_size=3, padding=1)
            self.y5 = nn.Conv2d(in_channels=self.filters[-1], out_channels=self.num_classes, kernel_size=3, padding=1)
        else:
            self.y1 = nn.Conv2d(in_channels=self.filters[0], out_channels=self.num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder

        c1, p1 = self.e1(x)
        c2, p2 = self.e2(p1)
        c3, p3 = self.e3(p2)
        c4, p4 = self.e4(p3)

        # Bottleneck / Last Encoder element

        bottleneck = self.bottleneck(p4)

        # CGM

        # cls = self.cgm(bottleneck)
        # cls_output = cls
        # cls = torch.argmax(cls)
        # cls = cls.view(cls.shape[0], 1, cls.shape[1], cls.shape[2])

        # Decoder 4

        e1_d4 = F.max_pool2d(c1, kernel_size=8, stride=8)
        e1_d4 = self.e1_d4(e1_d4)

        e2_d4 = F.max_pool2d(c2, kernel_size=4, stride=4)
        e2_d4 = self.e2_d4(e2_d4)

        e3_d4 = F.max_pool2d(c3, kernel_size=2, stride=2)
        e3_d4 = self.e3_d4(e3_d4)

        e4_d4 = self.e4_d4(c4)

        e5_d4 = F.interpolate(bottleneck, scale_factor=2, mode="bilinear", align_corners=True)
        e5_d4 = self.e5_d4(e5_d4)

        d4 = torch.cat([e1_d4, e2_d4, e3_d4, e4_d4, e5_d4], dim=1)
        d4 = self.d4(d4)

        # Decoder 3

        e1_d3 = F.max_pool2d(c1, kernel_size=4, stride=4)
        e1_d3 = self.e1_d3(e1_d3)

        e2_d3 = F.max_pool2d(c2, kernel_size=2, stride=2)
        e2_d3 = self.e2_d3(e2_d3)

        e3_d3 = self.e3_d3(c3)

        e4_d3 = F.interpolate(d4, scale_factor=2, mode="bilinear", align_corners=True)
        e4_d3 = self.e4_d3(e4_d3)

        e5_d3 = F.interpolate(bottleneck, scale_factor=4, mode="bilinear", align_corners=True)
        e5_d3 = self.e5_d3(e5_d3)

        d3 = torch.concat([e1_d3, e2_d3, e3_d3, e4_d3, e5_d3], dim=1)
        d3 = self.d3(d3)

        # Decoder 2

        e1_d2 = F.max_pool2d(c1, kernel_size=2, stride=2)
        e1_d2 = self.e1_d2(e1_d2)

        e2_d2 = self.e2_d2(c2)

        e3_d2 = F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=True)
        e3_d2 = self.e3_d2(e3_d2)

        e4_d2 = F.interpolate(d4, scale_factor=4, mode="bilinear", align_corners=True)
        e4_d2 = self.e4_d2(e4_d2)

        e5_d2 = F.interpolate(bottleneck, scale_factor=8, mode="bilinear", align_corners=True)
        e5_d2 = self.e5_d2(e5_d2)

        d2 = torch.concat([e1_d2, e2_d2, e3_d2, e4_d2, e5_d2], dim=1)
        d2 = self.d2(d2)

        # Decoder 1

        e1_d1 = self.e1_d1(c1)

        e2_d1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=True)
        e2_d1 = self.e2_d1(e2_d1)

        e3_d1 = F.interpolate(d3, scale_factor=4, mode="bilinear", align_corners=True)
        e3_d1 = self.e3_d1(e3_d1)

        e4_d1 = F.interpolate(d4, scale_factor=8, mode="bilinear", align_corners=True)
        e4_d1 = self.e4_d1(e4_d1)

        e5_d1 = F.interpolate(bottleneck, scale_factor=16, mode="bilinear", align_corners=True)
        e5_d1 = self.e5_d1(e5_d1)

        d1 = torch.concat([e1_d1, e2_d1, e3_d1, e4_d1, e5_d1], dim=1)
        d1 = self.d1(d1)

        if self.deep_sup:
            y1 = self.y1(d1)
            y2 = F.interpolate(self.y2(d2), scale_factor=2, mode="bilinear", align_corners=True)
            y3 = F.interpolate(self.y3(d3), scale_factor=4, mode="bilinear", align_corners=True)
            y4 = F.interpolate(self.y4(d4), scale_factor=8, mode="bilinear", align_corners=True)
            y5 = F.interpolate(self.y5(bottleneck), scale_factor=16, mode="bilinear", align_corners=True)

            return [y1, y2, y3, y4, y5]
        else:
            y1 = self.y1(d1)

            return y1


if __name__ == "__main__":
    x = torch.rand((1, 3, 256, 256))
    model = Unet3Plus(3, num_classes=5)
    y1, y2, y3, y4, y5 = model(x)

    print(y1.shape)
    print(y2.shape)
    print(y3.shape)
    print(y4.shape)
    print(y5.shape)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of Parameters : {total_params}")

