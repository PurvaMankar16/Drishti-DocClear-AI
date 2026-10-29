from dataclasses import dataclass


CLASS_NAMES = ["blur", "dark", "low_contrast", "noise", "skew", "low_resolution"]


@dataclass
class TrainingConfig:
    seed: int = 42
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 3
    num_workers: int = 0
    pretrained: bool = True
    dataset_version: str = "synthetic-docclear-v1"
