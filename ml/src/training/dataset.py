from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .config import CLASS_NAMES


class DocumentQualityDataset(Dataset):
    def __init__(self, metadata: pd.DataFrame, ml_root: Path, transform=None):
        self.metadata = metadata.reset_index(drop=True)
        self.ml_root = Path(ml_root)
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        image = Image.open(self.ml_root / row.image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        labels = torch.tensor([float(row[name]) for name in CLASS_NAMES], dtype=torch.float32)
        return image, labels
