"""
dataset.py — Loads domain-specific manufacturing defect images for transfer learning.

Expected data layout (CSV-driven, so it works whether images come from a
folder-per-class dump or a MES/vision-system image export):

    data/
      images/
        img_00001.jpg
        img_00002.jpg
        ...
      annotations.csv

annotations.csv columns:
    filename, defect_type, severity, bbox_x, bbox_y, bbox_w, bbox_h,
    shift, machine_id, batch_id, camera_id, timestamp

  - defect_type   : one of model.DEFECT_CLASSES (use "ok" for good parts —
                    you need negative examples, ideally 3-5x the positive
                    count, since defects are rare events on a healthy line)
  - severity      : 0.0 for "ok", else a human-annotated 0-1 score. If your
                    QC team scores defects on a 1-5 or A/B/C scale, map it
                    to [0,1] with SEVERITY_MAP below rather than re-annotating.
  - bbox_*        : normalized [0,1] box around the defect region. Leave
                    blank (NaN) for "ok" images or if you skip localization.
  - shift/machine_id/batch_id : metadata carried through to the analytics
                    DB so the dashboard can slice defect rates by source.

If you don't have annotations yet, `scripts/bootstrap_labeling.py` (not
included here, but straightforward to build) can pre-cluster unlabeled
images by visual similarity to speed up manual labeling.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from constants import DEFECT_CLASSES

CLASS_TO_IDX = {c: i for i, c in enumerate(DEFECT_CLASSES)}

# Common ordinal QC scales -> continuous severity score.
SEVERITY_MAP = {
    "A": 0.15, "B": 0.45, "C": 0.75, "D": 0.95,   # letter grade scales
    1: 0.1, 2: 0.3, 3: 0.5, 4: 0.7, 5: 0.9,          # 1-5 scales
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int = 224, train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(8),
            # Mild color jitter only — real defects like "color_inconsistency"
            # ARE the signal, so don't jitter so hard you wash that out.
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class DefectDataset(Dataset):
    def __init__(self, csv_path: str, image_dir: str, image_size: int = 224, train: bool = True):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = build_transforms(image_size, train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        label = CLASS_TO_IDX[row["defect_type"]]
        severity = float(row.get("severity", 0.0) or 0.0)

        bbox = torch.tensor([
            row.get("bbox_x", 0.0) or 0.0,
            row.get("bbox_y", 0.0) or 0.0,
            row.get("bbox_w", 0.0) or 0.0,
            row.get("bbox_h", 0.0) or 0.0,
        ], dtype=torch.float32)

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "severity": torch.tensor(severity, dtype=torch.float32),
            "bbox": bbox,
        }

    def class_weights(self):
        """Inverse-frequency weights for CrossEntropyLoss — defect classes
        are almost always heavily imbalanced against 'ok'."""
        counts = self.df["defect_type"].value_counts()
        weights = torch.ones(len(DEFECT_CLASSES))
        for cls, idx in CLASS_TO_IDX.items():
            n = counts.get(cls, 0)
            weights[idx] = 1.0 / n if n > 0 else 0.0
        return weights / weights.sum() * len(DEFECT_CLASSES)
