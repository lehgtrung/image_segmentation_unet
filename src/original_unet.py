# full assembly of the sub-parts to form the complete net
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from PIL import Image
from torch.nn.functional import sigmoid


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=0, stride=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvDown, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        pool_x = self.pool(x)
        return pool_x, x


class ConvUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvUp, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1_dim = x1.size()[2]
        x2 = extract_img(x1_dim, x2)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.conv(x1)
        return x1


def extract_img(size, in_tensor):
    dim1, dim2 = in_tensor.size()[2:]
    in_tensor = in_tensor[:, :, int((dim1-size)/2):int((dim1+size)/2),
                          int((dim2-size)/2):int((dim2+size)/2)]
    return in_tensor


class OriginalUnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OriginalUnet, self).__init__()
        self.Conv_down1 = ConvDown(in_channels, 64)
        self.Conv_down2 = ConvDown(64, 128)
        self.Conv_down3 = ConvDown(128, 256)
        self.Conv_down4 = ConvDown(256, 512)
        self.Conv_down5 = ConvDown(512, 1024)
        self.Conv_up1 = ConvUp(1024, 512)
        self.Conv_up2 = ConvUp(512, 256)
        self.Conv_up3 = ConvUp(256, 128)
        self.Conv_up4 = ConvUp(128, 64)
        self.Conv_out = nn.Conv2d(64, out_channels, 1, padding=0, stride=1)

    def forward(self, x):

        x, conv1 = self.Conv_down1(x)
        x, conv2 = self.Conv_down2(x)
        x, conv3 = self.Conv_down3(x)
        x, conv4 = self.Conv_down4(x)
        _, x = self.Conv_down5(x)
        x = self.Conv_up1(x, conv4)
        x = self.Conv_up2(x, conv3)
        x = self.Conv_up3(x, conv2)
        x = self.Conv_up4(x, conv1)
        x = self.Conv_out(x)
        return x


