import torch
import torch.nn as nn
import torch.nn.functional as F


class StackEncoder(nn.Module):
    def __init__(self, x_channels, y_channels, kernel_size=(3, 3)):
        super(StackEncoder, self).__init__()
        padding = (kernel_size - 1) // 2
        self.encode = nn.Sequential(
            nn.Conv2d(x_channels, y_channels, kernel_size=kernel_size, padding=padding),
            nn.Conv2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, x):
        x = self.encode(x)
        x_small = F.max_pool2d(x, kernel_size=2, stride=2)
        return x, x_small


class StackDecoder(nn.Module):
    def __init__(self, x_big_channels, x_channels, y_channels, kernel_size=3):
        super(StackDecoder, self).__init__()
        padding = (kernel_size - 1) // 2

        self.decode = nn.Sequential(
            nn.Conv2d(x_big_channels + x_channels, y_channels, kernel_size=kernel_size, padding=padding),
            nn.Conv2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding),
            nn.Conv2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, x, down_tensor):
        _, channels, height, width = down_tensor.size()
        x = F.interpolate(x, size=(height, width), mode='bilinear', align_corners=False)
        x = torch.cat([x, down_tensor], 1)
        x = self.decode(x)
        return x


# 1024x1024
class UNet1024(nn.Module):
    def __init__(self, in_shape):
        super(UNet1024, self).__init__()
        channels, height, width = in_shape

        # 1024
        self.down1 = StackEncoder(channels, 24, kernel_size=3)  # 512
        self.down2 = StackEncoder(24, 64, kernel_size=3)  # 256
        self.down3 = StackEncoder(64, 128, kernel_size=3)  # 128
        self.down4 = StackEncoder(128, 256, kernel_size=3)  # 64
        self.down5 = StackEncoder(256, 512, kernel_size=3)  # 32
        self.down6 = StackEncoder(512, 768, kernel_size=3)  # 16

        self.center = nn.Sequential(
            nn.Conv2d(768, 768, kernel_size=3, padding=1),
        )

        # 8
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(768, 768, 512, kernel_size=3)  # 16
        self.up5 = StackDecoder(512, 512, 256, kernel_size=3)  # 32
        self.up4 = StackDecoder(256, 256, 128, kernel_size=3)  # 64
        self.up3 = StackDecoder(128, 128, 64, kernel_size=3)  # 128
        self.up2 = StackDecoder(64, 64, 24, kernel_size=3)  # 256
        self.up1 = StackDecoder(24, 24, 24, kernel_size=3)  # 512
        self.classify = nn.Conv2d(24, 1, kernel_size=1, bias=True)

    def forward(self, x):
        out = x
        down1, out = self.down1(out)
        down2, out = self.down2(out)
        down3, out = self.down3(out)
        down4, out = self.down4(out)
        down5, out = self.down5(out)
        down6, out = self.down6(out)

        out = self.center(out)
        out = self.up6(out, down6)
        out = self.up5(out, down5)
        out = self.up4(out, down4)
        out = self.up3(out, down3)
        out = self.up2(out, down2)
        out = self.up1(out, down1)

        out = self.classify(out)
        out = torch.squeeze(out, dim=1)
        return out

