"""Construction helpers for the released IF2F architecture."""

import torch

from models.e4e import load_e4e_encoder
from models.encoders.model_irse import Backbone
from models.multi_scale_conditional_latent_transformer import (
    TransformerMapperSplitMulti,
)
from models.robust_face_hashing import SimpleKeyGenerator
from models.stylegan2.model import Generator


IMAGE_SIZE = 256


def build_mapper(device):
    return TransformerMapperSplitMulti(
        block_num=1,
        split_list=[4, 4, 6],
        normalize_type="layernorm",
        add_linear=True,
        add_pos_embedding=True,
    ).to(device)


def load_stylegan(checkpoint_path, device):
    generator = Generator(IMAGE_SIZE, 512, 8).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    generator.load_state_dict(checkpoint["g_ema"], strict=False)
    return generator.eval().requires_grad_(False)


def load_face_backbone(checkpoint_path, device):
    backbone = Backbone(
        input_size=112,
        num_layers=50,
        drop_ratio=0.6,
        mode="ir_se",
    ).to(device)
    backbone.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    return backbone.eval().requires_grad_(False)


def build_key_generator(backbone, device):
    return SimpleKeyGenerator(
        size=IMAGE_SIZE,
        backbone=backbone,
        requires_grad=False,
    ).to(device)


def load_inference_components(paths, device):
    checkpoint = torch.load(paths.if2f, map_location="cpu")
    encoder = load_e4e_encoder(paths.e4e, device)
    generator = load_stylegan(paths.stylegan, device)
    backbone = load_face_backbone(paths.ir_se50, device)
    key_generator = build_key_generator(backbone, device)
    mapper = build_mapper(device)
    mapper.load_state_dict(checkpoint["mapper_state_dict"])
    key_generator.load_state_dict(checkpoint["key_generator_state_dict"])
    del checkpoint
    mapper.eval()
    key_generator.eval()
    return encoder, generator, mapper, key_generator
