"""Create the CelebA-HQ metadata files required by public IF2F training."""

import argparse
from pathlib import Path


SUPPORTED_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def collect_identities(image_root):
    identities = []
    for identity_dir in sorted(path for path in image_root.iterdir() if path.is_dir()):
        if any(character.isspace() for character in identity_dir.name):
            raise ValueError(
                f"Whitespace is not supported in identity names: {identity_dir}"
            )
        images = sorted(
            path
            for path in identity_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if len(images) >= 2:
            for image in images:
                if any(character.isspace() for character in image.name):
                    raise ValueError(
                        f"Whitespace is not supported in metadata paths: {image}"
                    )
            identities.append((identity_dir.name, images))
    return identities


def relative_path(identity_name, image_path):
    return (Path(identity_name) / image_path.name).as_posix()


def write_lines(path, lines, overwrite):
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {path}. Re-run with --overwrite if intended."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_training_pairs(identities):
    lines = [str(len(identities))]
    for label, (identity_name, images) in enumerate(identities):
        for index, image_a in enumerate(images):
            image_b = images[(index + 1) % len(images)]
            lines.append(
                f"{relative_path(identity_name, image_a)} "
                f"{relative_path(identity_name, image_b)} {label}"
            )
    return lines


def build_validation_triplets(identities):
    lines = [str(len(identities))]
    for index in range(len(identities)):
        identity_name, images = identities[index]
        negative_name, negative_images = identities[(index + 1) % len(identities)]
        image_a, image_b = images[0], images[1]
        for negative in negative_images[:2]:
            lines.append(
                f"{relative_path(identity_name, image_a)} "
                f"{relative_path(identity_name, image_b)} "
                f"{relative_path(negative_name, negative)}"
            )
    return lines


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare identity pairs and validation triplets for IF2F."
    )
    parser.add_argument("--data_root", required=True, help="CelebA-HQ dataset root.")
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--validation_fraction", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 < args.train_fraction < 1:
        parser.error("--train_fraction must be between 0 and 1")
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation_fraction must be between 0 and 1")
    if args.train_fraction + args.validation_fraction >= 1:
        parser.error("Training and validation fractions must sum to less than 1")
    return args


def main():
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    image_root = data_root / "CelebA-HQ_classified"
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_root}")

    identities = collect_identities(image_root)
    if len(identities) < 10:
        raise ValueError(
            "At least 10 identities with two images each are required to create splits."
        )

    train_end = int(len(identities) * args.train_fraction)
    validation_end = train_end + int(len(identities) * args.validation_fraction)
    training_identities = identities[:train_end]
    validation_identities = identities[train_end:validation_end]
    if len(validation_identities) < 2:
        raise ValueError("The validation split must contain at least two identities.")

    pair_path = data_root / "pairs_celeba-hq_train.txt"
    triplet_path = data_root / "triples_celeba-hq_val_nolabel.txt"
    if not args.overwrite:
        existing = [path for path in (pair_path, triplet_path) if path.exists()]
        if existing:
            joined = ", ".join(str(path) for path in existing)
            raise FileExistsError(
                f"Refusing to overwrite existing metadata: {joined}. "
                "Re-run with --overwrite if intended."
            )
    write_lines(
        pair_path,
        build_training_pairs(training_identities),
        args.overwrite,
    )
    write_lines(
        triplet_path,
        build_validation_triplets(validation_identities),
        args.overwrite,
    )

    print(f"Eligible identities: {len(identities)}")
    print(f"Training identities: {len(training_identities)} -> {pair_path}")
    print(f"Validation identities: {len(validation_identities)} -> {triplet_path}")


if __name__ == "__main__":
    main()
