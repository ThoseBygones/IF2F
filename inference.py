"""Generate IF2F anonymized faces from one image or a directory."""

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from models.if2f import IMAGE_SIZE, load_inference_components
from utils import is_within, require_files


PROJECT_ROOT = Path(__file__).resolve().parent
SUPPORTED_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class InputImages(Dataset):
    def __init__(self, input_path, exclude_path=None):
        self.input_path = Path(input_path).expanduser().resolve()
        excluded = Path(exclude_path).resolve() if exclude_path is not None else None

        if self.input_path.is_file():
            if self.input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError(f"Unsupported image type: {self.input_path.suffix}")
            self.root = self.input_path.parent
            self.paths = [self.input_path]
        elif self.input_path.is_dir():
            self.root = self.input_path
            self.paths = sorted(
                path
                for path in self.input_path.rglob("*")
                if path.is_file()
                and path.suffix.lower() in SUPPORTED_EXTENSIONS
                and (excluded is None or not is_within(path, excluded))
            )
        else:
            raise FileNotFoundError(f"Input path not found: {self.input_path}")

        if not self.paths:
            raise ValueError(f"No supported images found in: {self.input_path}")

        self.transform = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), path.relative_to(self.root).as_posix()


def output_path(root, relative_path):
    path = root / Path(relative_path).with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def image_for_saving(image):
    return ((image + 1) / 2).clamp(0, 1)


class IF2FInference:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        resolved = require_files(
            if2f=args.if2f_path,
            e4e=args.e4e_path,
            stylegan=args.stylegan_path,
            ir_se50=args.ir_se50_path,
        )
        model_paths = argparse.Namespace(**resolved)
        (
            self.encoder,
            self.generator,
            self.mapper,
            self.key_generator,
        ) = load_inference_components(model_paths, self.device)

    @torch.no_grad()
    def run(self):
        output_root = Path(self.args.output_dir).expanduser().resolve()
        dataset = InputImages(self.args.input, exclude_path=output_root)
        loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.workers,
            pin_memory=self.device.type == "cuda",
        )

        original_root = output_root / "original"
        anonymized_root = output_root / "anonymized"
        rows = []
        similarities = []

        for images, relative_paths in tqdm(loader, desc="Anonymizing"):
            images = images.to(self.device, non_blocking=True)
            original_latents = self.encoder(images)
            hash_conditions = self.key_generator(images)
            anonymized_latents = self.mapper(
                torch.cat([original_latents, hash_conditions], dim=-1)
            )
            anonymized, _ = self.generator(
                [anonymized_latents],
                input_is_latent=True,
                randomize_noise=False,
                truncation=1,
            )

            original_features = hash_conditions[:, 0]
            anonymized_features = self.key_generator.extract_feats(anonymized)
            batch_similarities = F.cosine_similarity(
                original_features, anonymized_features
            ).detach().cpu()

            for index, relative_path in enumerate(relative_paths):
                original_file = output_path(original_root, relative_path)
                anonymized_file = output_path(anonymized_root, relative_path)
                save_image(image_for_saving(images[index]), original_file)
                save_image(image_for_saving(anonymized[index]), anonymized_file)

                similarity = float(batch_similarities[index])
                similarities.append(similarity)
                rows.append(
                    {
                        "input": relative_path,
                        "original_output": original_file.relative_to(output_root).as_posix(),
                        "anonymized_output": anonymized_file.relative_to(output_root).as_posix(),
                        "cosine_similarity": f"{similarity:.8f}",
                    }
                )

        output_root.mkdir(parents=True, exist_ok=True)
        csv_path = output_root / "cosine_similarity.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "input",
                    "original_output",
                    "anonymized_output",
                    "cosine_similarity",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        similarity_tensor = torch.tensor(similarities)
        print(f"Processed {len(dataset)} image(s). Results: {output_root}")
        print(
            "Original/anonymized identity cosine similarity: "
            f"mean={similarity_tensor.mean():.6f}, "
            f"min={similarity_tensor.min():.6f}, "
            f"max={similarity_tensor.max():.6f}"
        )
        print(f"Per-image similarities: {csv_path}")


def parse_args():
    pretrained = PROJECT_ROOT / "pretrained_models"
    parser = argparse.ArgumentParser(
        description="Generate anonymized face images with the released IF2F model."
    )
    parser.add_argument("--input", required=True, help="Input image or directory.")
    parser.add_argument("--output_dir", default="./outputs", help="Output directory.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--if2f_path", default=str(pretrained / "best_iter212000.pt")
    )
    parser.add_argument(
        "--e4e_path", default=str(pretrained / "e4e_ffhq_encode_256.pt")
    )
    parser.add_argument(
        "--stylegan_path", default=str(pretrained / "stylegan2-ffhq-256.pt")
    )
    parser.add_argument(
        "--ir_se50_path", default=str(pretrained / "model_ir_se50.pth")
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch_size must be at least 1")
    if args.workers < 0:
        parser.error("--workers cannot be negative")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable; pass --device cpu to continue.")
    return args


def main():
    IF2FInference(parse_args()).run()


if __name__ == "__main__":
    main()
