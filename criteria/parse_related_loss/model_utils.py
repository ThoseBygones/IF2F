"""UNet blocks required by the IF2F face parsing loss."""

import torch
import torch.nn.functional as F
from torch import nn


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super().__init__()
        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, 1, 1),
                nn.BatchNorm2d(out_size),
                nn.ReLU(),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, 3, 1, 1),
                nn.BatchNorm2d(out_size),
                nn.ReLU(),
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, 1, 1), nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, 3, 1, 1), nn.ReLU()
            )

    def forward(self, inputs):
        return self.conv2(self.conv1(inputs))


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, is_batchnorm):
        super().__init__()
        self.conv = unetConv2(in_size, out_size, is_batchnorm)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, skip, inputs):
        inputs = self.up(inputs)
        offset = inputs.size(2) - skip.size(2)
        padding = 2 * [offset // 2, offset // 2]
        skip = F.pad(skip, padding)
        return self.conv(torch.cat([skip, inputs], dim=1))
