"""Load the frozen e4e encoder used by IF2F."""

from argparse import Namespace

import torch

from models.encoders import psp_encoders


def _encoder_state(checkpoint):
    state_dict = checkpoint.get("state_dict", checkpoint)
    prefix = "encoder."
    return {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


def _build_encoder(options):
    encoder_type = options.encoder_type
    if encoder_type == "GradualStyleEncoder":
        return psp_encoders.GradualStyleEncoder(50, "ir_se", options)
    if encoder_type == "Encoder4Editing":
        return psp_encoders.Encoder4Editing(50, "ir_se", options)
    if encoder_type == "SingleStyleCodeEncoder":
        return psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, "ir_se", options)
    raise ValueError(f"Unsupported e4e encoder type: {encoder_type}")


def load_e4e_encoder(checkpoint_path, device):
    """Load only the encoder from an e4e/pSp checkpoint.

    IF2F uses a separately configured StyleGAN2 generator, so constructing the
    complete pSp model would allocate an unused second generator.
    """

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_options = checkpoint["opts"]
    if isinstance(checkpoint_options, Namespace):
        checkpoint_options = vars(checkpoint_options)
    options = Namespace(**dict(checkpoint_options))

    encoder = _build_encoder(options)
    encoder.load_state_dict(_encoder_state(checkpoint), strict=False)
    encoder.eval().requires_grad_(False)
    return encoder.to(device)
