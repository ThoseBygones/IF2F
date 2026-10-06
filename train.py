"""Train the released IF2F model on identity-classified CelebA-HQ images."""

import argparse
import json
import logging
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from criteria.id_loss import IDLossExtractor
from criteria.lpips.lpips import LPIPS
from criteria.parse_related_loss.bg_loss import ParseLoss
from datasets.celeba_hq import TrainingPairs, ValidationTriplets
from models.e4e import load_e4e_encoder
from models.feature_bank import FeatureBankPlus
from models.if2f import (
    IMAGE_SIZE,
    build_key_generator,
    build_mapper,
    load_face_backbone,
    load_stylegan,
)
from training.ranger import Ranger
from utils import require_files


PROJECT_ROOT = Path(__file__).resolve().parent


def configure_logging(experiment_dir):
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("if2f")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(experiment_dir / "record.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


class IF2FTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.experiment_dir = Path(args.exp_dir).expanduser().resolve()
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.image_dir = self.experiment_dir / "images"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.image_dir.mkdir(exist_ok=True)
        self.logger = configure_logging(self.experiment_dir)
        self.writer = SummaryWriter(log_dir=str(self.experiment_dir / "logs"))

        transform = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )
        self.train_dataset = TrainingPairs(args.data_root, transform)
        self.validation_dataset = ValidationTriplets(args.data_root, transform)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            drop_last=True,
            pin_memory=self.device.type == "cuda",
        )
        self.validation_loader = DataLoader(
            self.validation_dataset,
            batch_size=args.validation_batch_size,
            shuffle=False,
            num_workers=args.validation_workers,
            drop_last=False,
            pin_memory=self.device.type == "cuda",
        )
        self.train_iterator = iter(self.train_loader)

        self.encoder = load_e4e_encoder(args.e4e_path, self.device)
        self.generator = load_stylegan(args.stylegan_path, self.device)

        key_backbone = load_face_backbone(args.ir_se50_path, self.device)
        self.key_generator = build_key_generator(key_backbone, self.device)
        self.mapper = build_mapper(self.device)

        loss_backbone = load_face_backbone(args.ir_se50_path, self.device)
        self.identity_loss = IDLossExtractor(
            loss_backbone, margin=args.identity_margin
        ).to(self.device)
        self.lpips_loss = LPIPS(net_type="vgg").to(self.device).eval()
        self.parse_loss = ParseLoss(args.parsenet_path, IMAGE_SIZE).to(self.device).eval()
        self.contrastive_loss = nn.CrossEntropyLoss().to(self.device)
        self.feature_bank = FeatureBankPlus(
            num_identities=self.train_dataset.num_classes,
            queue_length=2,
            temperature=0.07,
        ).to(self.device)
        self.face_pool = nn.AdaptiveAvgPool2d((IMAGE_SIZE, IMAGE_SIZE)).to(self.device)

        parameter_groups = [
            {"params": self.mapper.parameters()},
            {"params": self.key_generator.parameters()},
        ]
        if args.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                parameter_groups, lr=args.learning_rate
            )
        else:
            self.optimizer = Ranger(parameter_groups, lr=args.learning_rate)

        self.global_step = 0
        self.best_eer = math.inf
        if args.resume is not None:
            self.load_checkpoint(args.resume)

        self.logger.info("Training samples: %d", len(self.train_dataset))
        self.logger.info("Validation triplets: %d", len(self.validation_dataset))
        self.logger.info("Training identities: %d", self.train_dataset.num_classes)

    def next_batch(self):
        try:
            return next(self.train_iterator)
        except StopIteration:
            self.train_iterator = iter(self.train_loader)
            return next(self.train_iterator)

    def anonymize(self, image):
        original_latent = self.encoder(image)
        hash_condition = self.key_generator(image)
        anonymized_latent = self.mapper(
            torch.cat([original_latent, hash_condition], dim=-1)
        )
        anonymized, _ = self.generator(
            [anonymized_latent],
            input_is_latent=True,
            randomize_noise=False,
            truncation=1,
        )
        return anonymized

    def calculate_loss(
        self,
        image_a,
        image_b,
        generated_a,
        generated_b,
        labels,
    ):
        losses = {}
        total = torch.zeros((), device=self.device)

        lpips = self.lpips_loss(image_a, generated_a)
        lpips = lpips + self.lpips_loss(image_b, generated_b)
        losses["lpips"] = lpips * self.args.lambda_lpips
        total = total + losses["lpips"]

        anonymity_a = self.identity_loss(image_a, generated_a)
        anonymity_b = self.identity_loss(image_b, generated_b)
        anonymity = (anonymity_a + anonymity_b) / 2
        losses["anonymity"] = anonymity * self.args.lambda_anonymity
        total = total + losses["anonymity"]

        original_features_a = self.identity_loss.extract_feats(image_a)
        original_features_b = self.identity_loss.extract_feats(image_b)
        generated_features_a = self.identity_loss.extract_feats(generated_a)
        generated_features_b = self.identity_loss.extract_feats(generated_b)
        logits, targets = self.feature_bank(
            generated_features_a,
            generated_features_b,
            original_features_a,
            original_features_b,
            labels,
        )
        contrastive = self.contrastive_loss(logits, targets)
        losses["contrastive"] = contrastive * self.args.lambda_contrastive
        total = total + losses["contrastive"]

        parse = self.parse_loss(image_a, generated_a)
        parse = parse + self.parse_loss(image_b, generated_b)
        losses["parse"] = parse * self.args.lambda_parse
        total = total + losses["parse"]

        losses["total"] = total
        return total, losses

    def train_step(self):
        self.mapper.train()
        self.key_generator.eval()
        self.optimizer.zero_grad()

        image_a, image_b, labels = self.next_batch()
        image_a = image_a.to(self.device, non_blocking=True)
        image_b = image_b.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        with torch.no_grad():
            latent_a = self.encoder(image_a)
            latent_b = self.encoder(image_b)
            reconstructed_a, _ = self.generator(
                [latent_a], input_is_latent=True, randomize_noise=False, truncation=1
            )
            reconstructed_b, _ = self.generator(
                [latent_b], input_is_latent=True, randomize_noise=False, truncation=1
            )
            hash_a = self.key_generator(image_a)
            hash_b = self.key_generator(image_b)

        anonymized_latent_a = self.mapper(torch.cat([latent_a, hash_a], dim=-1))
        anonymized_latent_b = self.mapper(torch.cat([latent_b, hash_b], dim=-1))
        generated_a, _ = self.generator(
            [anonymized_latent_a],
            input_is_latent=True,
            randomize_noise=False,
            truncation=1,
        )
        generated_b, _ = self.generator(
            [anonymized_latent_b],
            input_is_latent=True,
            randomize_noise=False,
            truncation=1,
        )

        reconstructed_a = self.face_pool(reconstructed_a)
        reconstructed_b = self.face_pool(reconstructed_b)
        generated_a = self.face_pool(generated_a)
        generated_b = self.face_pool(generated_b)

        total, losses = self.calculate_loss(
            reconstructed_a,
            reconstructed_b,
            generated_a,
            generated_b,
            labels,
        )
        total.backward()
        self.optimizer.step()
        return losses, (reconstructed_a, generated_a, reconstructed_b, generated_b)

    @torch.no_grad()
    def validate(self):
        self.mapper.eval()
        self.key_generator.eval()
        positive_scores = []
        negative_scores = []

        for batch_index, (image_a, image_b, negative) in enumerate(
            tqdm(self.validation_loader, desc="Validating", leave=False)
        ):
            image_a = image_a.to(self.device, non_blocking=True)
            image_b = image_b.to(self.device, non_blocking=True)
            negative = negative.to(self.device, non_blocking=True)

            generated = []
            for image in (image_a, image_b, negative):
                generated.append(self.face_pool(self.anonymize(image)))

            features_a = self.identity_loss.extract_feats(generated[0])
            features_b = self.identity_loss.extract_feats(generated[1])
            features_negative = self.identity_loss.extract_feats(generated[2])
            positive_scores.append(F.cosine_similarity(features_a, features_b).clamp_min(0))
            negative_scores.append(
                F.cosine_similarity(features_b, features_negative).clamp_min(0)
            )

            if batch_index < self.args.validation_image_batches:
                grid = torch.cat(
                    [image_a, generated[0], image_b, generated[1], negative, generated[2]],
                    dim=0,
                )
                save_image(
                    grid,
                    self.image_dir / f"validation_{self.global_step:06d}_{batch_index}.png",
                    nrow=image_a.shape[0],
                    normalize=True,
                    value_range=(-1, 1),
                )

        positive_scores = torch.cat(positive_scores)
        negative_scores = torch.cat(negative_scores)
        thresholds = torch.arange(100, device=self.device) / 100
        true_positive_rates = torch.stack(
            [(positive_scores >= threshold).float().mean() for threshold in thresholds]
        )
        false_positive_rates = torch.stack(
            [(negative_scores >= threshold).float().mean() for threshold in thresholds]
        )
        index = torch.argmin(
            torch.abs(1 - true_positive_rates - false_positive_rates)
        )
        eer = (1 - true_positive_rates[index] + false_positive_rates[index]) / 2
        auc = metrics.auc(
            false_positive_rates.cpu().numpy(), true_positive_rates.cpu().numpy()
        )
        return float(eer), float(auc), float(thresholds[index])

    def log_losses(self, losses):
        values = {name: float(value.detach()) for name, value in losses.items()}
        message = ", ".join(f"{name}={value:.6f}" for name, value in values.items())
        self.logger.info("step=%d | %s", self.global_step, message)
        for name, value in values.items():
            self.writer.add_scalar(f"train/{name}", value, self.global_step)

    def save_training_images(self, images):
        save_image(
            torch.cat(images, dim=0),
            self.image_dir / f"train_{self.global_step:06d}.png",
            nrow=images[0].shape[0],
            normalize=True,
            value_range=(-1, 1),
        )

    def checkpoint_state(self):
        return {
            "mapper_state_dict": self.mapper.state_dict(),
            "key_generator_state_dict": self.key_generator.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "best_eer": self.best_eer,
            "opts": vars(self.args),
        }

    def save_checkpoint(self, name):
        path = self.checkpoint_dir / name
        torch.save(self.checkpoint_state(), path)
        self.logger.info("Saved checkpoint: %s", path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.mapper.load_state_dict(checkpoint["mapper_state_dict"])
        self.key_generator.load_state_dict(checkpoint["key_generator_state_dict"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.global_step = int(checkpoint.get("global_step", -1)) + 1
        self.best_eer = float(checkpoint.get("best_eer", math.inf))
        self.logger.info("Resumed from %s at step %d", checkpoint_path, self.global_step)

    def train(self):
        while self.global_step < self.args.max_steps:
            losses, images = self.train_step()

            if self.global_step % self.args.log_interval == 0:
                self.log_losses(losses)
            if self.global_step % self.args.image_interval == 0:
                self.save_training_images(images)

            if self.global_step > 0 and self.global_step % self.args.validation_interval == 0:
                eer, auc, threshold = self.validate()
                self.logger.info(
                    "validation step=%d | EER=%.6f, AUC=%.6f, threshold=%.2f",
                    self.global_step,
                    eer,
                    auc,
                    threshold,
                )
                self.writer.add_scalar("validation/eer", eer, self.global_step)
                self.writer.add_scalar("validation/auc", auc, self.global_step)
                if eer < self.best_eer:
                    self.best_eer = eer
                    self.save_checkpoint("best.pt")

            is_last = self.global_step + 1 == self.args.max_steps
            if (
                self.global_step > 0
                and self.global_step % self.args.save_interval == 0
            ) or is_last:
                self.save_checkpoint(f"iteration_{self.global_step}.pt")

            self.global_step += 1

        self.writer.close()
        self.logger.info("Training completed at step %d", self.global_step)


def parse_args():
    pretrained = PROJECT_ROOT / "pretrained_models"
    parser = argparse.ArgumentParser(description="Train the released IF2F model.")
    parser.add_argument("--data_root", required=True, help="CelebA-HQ dataset root.")
    parser.add_argument("--exp_dir", default="./experiments/if2f")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", help="Checkpoint used to resume training.")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--validation_batch_size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--validation_workers", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.0002)
    parser.add_argument("--optimizer", choices=["ranger", "adam"], default="ranger")
    parser.add_argument("--max_steps", type=int, default=300000)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--image_interval", type=int, default=500)
    parser.add_argument("--validation_interval", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=10000)
    parser.add_argument("--validation_image_batches", type=int, default=1)
    parser.add_argument("--identity_margin", type=float, default=0.01)
    parser.add_argument("--lambda_anonymity", type=float, default=0.5)
    parser.add_argument("--lambda_contrastive", type=float, default=0.1)
    parser.add_argument("--lambda_lpips", type=float, default=1.0)
    parser.add_argument("--lambda_parse", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--e4e_path", default=str(pretrained / "e4e_ffhq_encode_256.pt")
    )
    parser.add_argument(
        "--stylegan_path", default=str(pretrained / "stylegan2-ffhq-256.pt")
    )
    parser.add_argument(
        "--ir_se50_path", default=str(pretrained / "model_ir_se50.pth")
    )
    parser.add_argument(
        "--parsenet_path", default=str(pretrained / "parsenet.pth")
    )
    args = parser.parse_args()

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable; pass --device cpu to continue.")
    for name in (
        "batch_size",
        "validation_batch_size",
        "max_steps",
        "log_interval",
        "image_interval",
        "validation_interval",
        "save_interval",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be at least 1")
    if args.workers < 0 or args.validation_workers < 0:
        parser.error("Worker counts cannot be negative.")

    require_files(
        e4e=args.e4e_path,
        stylegan=args.stylegan_path,
        ir_se50=args.ir_se50_path,
        parsenet=args.parsenet_path,
    )
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    experiment_dir = Path(args.exp_dir).expanduser().resolve()
    experiment_dir.mkdir(parents=True, exist_ok=True)
    with (experiment_dir / "options.json").open("w", encoding="utf-8") as stream:
        json.dump(vars(args), stream, indent=2, sort_keys=True)

    IF2FTrainer(args).train()


if __name__ == "__main__":
    main()
