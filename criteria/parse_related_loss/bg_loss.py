"""Face parsing loss used by the released IF2F training configuration."""

import torch
import torch.nn.functional as F
from torch import nn

from criteria.parse_related_loss.unet import unet


class ParseLoss(nn.Module):
    def __init__(self, weights_path, image_size=256):
        super().__init__()
        self.parsenet = unet()
        self.parsenet.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.parsenet.eval().requires_grad_(False)
        self.shrink = nn.AdaptiveAvgPool2d((512, 512))
        self.magnify = nn.AdaptiveAvgPool2d((image_size, image_size))
        self.selected_classes = [2, 4, 5, 10, 11, 12]

    def predict(self, image):
        prediction = self.magnify(self.parsenet(self.shrink(image)))
        selection = torch.zeros_like(prediction)
        selection[:, self.selected_classes] = 1
        return selection * prediction

    def forward(self, image, generated):
        return F.mse_loss(self.predict(image), self.predict(generated))
