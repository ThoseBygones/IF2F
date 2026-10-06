"""CelebA-HQ datasets used by the public IF2F training pipeline."""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class _MetadataDataset(Dataset):
    def __init__(self, root, metadata_name, transform):
        self.root = Path(root).expanduser().resolve()
        self.image_root = self.root / "CelebA-HQ_classified"
        self.metadata_path = self.root / metadata_name
        self.transform = transform

        if not self.image_root.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_root}")
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        lines = [line.strip() for line in self.metadata_path.read_text().splitlines()]
        if not lines:
            raise ValueError(f"Metadata file is empty: {self.metadata_path}")
        self.num_classes = int(lines[0])
        self.records = [line for line in lines[1:] if line]
        if not self.records:
            raise ValueError(f"Metadata file has no samples: {self.metadata_path}")

    def _load(self, relative_path):
        path = self.image_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Image listed in metadata was not found: {path}")
        image = Image.open(path).convert("RGB")
        return self.transform(image)

    def __len__(self):
        return len(self.records)


class TrainingPairs(_MetadataDataset):
    def __init__(self, root, transform):
        super().__init__(root, "pairs_celeba-hq_train.txt", transform)

    def __getitem__(self, index):
        image_a, image_b, identity = self.records[index].split()
        return self._load(image_a), self._load(image_b), int(identity)


class ValidationTriplets(_MetadataDataset):
    def __init__(self, root, transform):
        super().__init__(root, "triples_celeba-hq_val_nolabel.txt", transform)

    def __getitem__(self, index):
        image_a, image_b, negative = self.records[index].split()
        return self._load(image_a), self._load(image_b), self._load(negative)
