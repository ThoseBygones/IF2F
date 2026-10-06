"""Check the public IF2F runtime and required model files."""

import argparse
import importlib
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEPENDENCIES = {
    "numpy": "NumPy",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "tensorboard": "TensorBoard",
    "torch": "PyTorch",
    "torchvision": "TorchVision",
    "tqdm": "tqdm",
}

PROJECT_IMPORTS = [
    "criteria.id_loss",
    "criteria.lpips.lpips",
    "criteria.parse_related_loss.bg_loss",
    "datasets.celeba_hq",
    "models.if2f",
    "models.multi_scale_conditional_latent_transformer",
    "models.robust_face_hashing",
]


def module_version(module):
    return getattr(module, "__version__", "available")


def parse_args():
    parser = argparse.ArgumentParser(description="Check an IF2F installation.")
    parser.add_argument(
        "--mode", choices=["inference", "training"], default="inference"
    )
    parser.add_argument("--allow_cpu", action="store_true")
    parser.add_argument(
        "--pretrained_dir", default=str(PROJECT_ROOT / "pretrained_models")
    )
    return parser.parse_args()


def main():
    args = parse_args()
    failures = []
    loaded = {}

    print("Dependency imports:")
    for import_name, display_name in DEPENDENCIES.items():
        try:
            module = importlib.import_module(import_name)
            loaded[import_name] = module
            print(f"  [OK] {display_name:<16} {module_version(module)}")
        except Exception as error:
            failures.append(f"{display_name}: {error}")
            print(f"  [FAIL] {display_name:<14} {error}")

    print("\nProject imports:")
    for import_name in PROJECT_IMPORTS:
        try:
            importlib.import_module(import_name)
            print(f"  [OK] {import_name}")
        except Exception as error:
            failures.append(f"{import_name}: {error}")
            print(f"  [FAIL] {import_name}: {error}")

    required = [
        "best_iter212000.pt",
        "e4e_ffhq_encode_256.pt",
        "model_ir_se50.pth",
        "stylegan2-ffhq-256.pt",
    ]
    if args.mode == "training":
        required.append("parsenet.pth")

    pretrained_dir = Path(args.pretrained_dir).expanduser().resolve()
    print("\nRequired model files:")
    for filename in required:
        path = pretrained_dir / filename
        if path.is_file():
            print(f"  [OK] {path}")
        else:
            failures.append(f"missing model file: {path}")
            print(f"  [FAIL] missing {path}")

    torch = loaded.get("torch")
    if torch is not None:
        print("\nCUDA:")
        print(f"  PyTorch build: {torch.version.cuda}")
        print(f"  Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  Device 0: {torch.cuda.get_device_name(0)}")
        elif not args.allow_cpu:
            failures.append("CUDA is unavailable; use --allow_cpu for a CPU-only check")

    if failures:
        print("\nEnvironment check failed:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)

    print("\nEnvironment check passed.")


if __name__ == "__main__":
    main()
