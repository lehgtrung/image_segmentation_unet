import torch
import torch.nn as nn
import torch.nn.functional as F


BN_EPS = 1e-4


class ConvBnRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=1):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=BN_EPS)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class StackEncoder(nn.Module):
    def __init__(self, x_channels, y_channels, kernel_size=(3, 3)):
        super(StackEncoder, self).__init__()
        padding = (kernel_size - 1) // 2
        self.encode = nn.Sequential(
            ConvBnRelu2d(x_channels, y_channels, kernel_size=kernel_size, padding=padding),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding),
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
            ConvBnRelu2d(x_big_channels + x_channels, y_channels, kernel_size=kernel_size, padding=padding),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding),
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
        self.down1 = StackEncoder(channels, 24, kernel_size=3)
        self.down2 = StackEncoder(24, 64, kernel_size=3)
        self.down3 = StackEncoder(64, 128, kernel_size=3)

        self.center = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
        )

        # 8
        # x_big_channels, x_channels, y_channels
        self.up3 = StackDecoder(128, 128, 64, kernel_size=3)
        self.up2 = StackDecoder(64, 64, 24, kernel_size=3)
        self.up1 = StackDecoder(24, 24, 24, kernel_size=3)
        self.classify = nn.Conv2d(24, 1, kernel_size=1, bias=True)

    def forward(self, x):
        out = x
        down1, out = self.down1(out)
        down2, out = self.down2(out)
        down3, out = self.down3(out)

        out = self.center(out)
        out = self.up3(out, down3)
        out = self.up2(out, down2)
        out = self.up1(out, down1)

        out = self.classify(out)
        out = torch.sigmoid(out)
        return out

