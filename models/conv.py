from torch import nn


class Conv(nn.Module):
    def __init__(self, in_C, out_C, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_C, out_channels=out_C, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_C),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
