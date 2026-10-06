"""Robust Face Hashing (RFH) module used by IF2F."""

import math

import torch
from torch import nn


class SimpleKeyGenerator(nn.Module):
    """Generate a repeated robust face-hash condition for StyleGAN latents."""

    def __init__(self, size, backbone, requires_grad=False):
        super().__init__()
        self.stylegan_size = size
        self.pretrained = not requires_grad
        self.latents_num = (int(math.log(size, 2)) - 1) * 2

        # This layer is not used by forward(), but is retained because it is
        # part of the released IF2F checkpoint state dictionary.
        self.mapping_linear = nn.Linear(512, 512 * self.latents_num)

        self.pool = nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = nn.AdaptiveAvgPool2d((112, 112))
        self.output_layer = backbone.output_layer
        self.input_layer = backbone.input_layer

        body = backbone.body
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for index in range(3):
            self.slice1.add_module(str(index), body[index])
        for index in range(3, 7):
            self.slice2.add_module(str(index), body[index])
        for index in range(7, 21):
            self.slice3.add_module(str(index), body[index])
        for index in range(21, 24):
            self.slice4.add_module(str(index), body[index])

        if self.pretrained:
            self.eval().requires_grad_(False)

    def extract_feats(self, image):
        if image.shape[-2:] != (256, 256):
            image = self.pool(image)
        feature = image[:, :, 35:223, 32:220]
        feature = self.face_pool(feature)
        feature = self.input_layer(feature)
        feature = self.slice1(feature)
        feature = self.slice2(feature)
        feature = self.slice3(feature)
        feature = self.slice4(feature)
        feature = self.output_layer(feature)
        return torch.nn.functional.normalize(feature, dim=1)

    def forward(self, image):
        if self.pretrained:
            with torch.no_grad():
                embedding = self.extract_feats(image)
        else:
            embedding = self.extract_feats(image)
        return embedding.unsqueeze(1).repeat(1, self.latents_num, 1)
