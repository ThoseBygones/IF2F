"""Identity features and anonymity loss used by IF2F training."""

import torch
import torch.nn.functional as F
from torch import nn


class IDLossExtractor(nn.Module):
    def __init__(self, backbone, margin=0.01):
        super().__init__()
        backbone.eval()
        self.margin = margin
        self.pool = nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = nn.AdaptiveAvgPool2d((112, 112))
        self.output_layer = backbone.output_layer
        self.input_layer = backbone.input_layer
        body = backbone.body
        self.slice1 = nn.Sequential(*body[0:3])
        self.slice2 = nn.Sequential(*body[3:7])
        self.slice3 = nn.Sequential(*body[7:21])
        self.slice4 = nn.Sequential(*body[21:24])
        self.eval().requires_grad_(False)

    def extract_feats(self, image):
        if image.shape[-2:] != (256, 256):
            image = self.pool(image)
        feature = self.face_pool(image[:, :, 35:223, 32:220])
        feature = self.input_layer(feature)
        for block in (self.slice1, self.slice2, self.slice3, self.slice4):
            feature = block(feature)
        feature = self.output_layer(feature)
        return F.normalize(feature, dim=1)

    def forward(self, image_a, image_b):
        features_a = self.extract_feats(image_a)
        features_b = self.extract_feats(image_b)
        similarity = torch.cosine_similarity(features_a, features_b)
        return similarity.clamp(min=self.margin).mean()
