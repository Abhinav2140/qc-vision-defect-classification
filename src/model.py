"""
model.py — Multi-task defect inspection CNN.

Backbone: transfer learning from EfficientNet-B0 or ResNet50 (ImageNet-pretrained),
with three heads trained jointly:

  1. defect_type   -> multi-class classification (softmax)
                      classes: {ok, scratch, dent, dimensional_error,
                                missing_component, color_inconsistency, contamination}
  2. severity       -> regression head, outputs a continuous score in [0, 1]
                      (0 = imperceptible, 1 = catastrophic)
  3. localization    -> auxiliary bounding-box regression (x, y, w, h), normalized
                      to [0,1], used to draw the defect region on the dashboard /
                      operator HMI. Optional — can be disabled if you don't have
                      box annotations yet (see `use_localization`).

Why a shared backbone with multiple heads instead of three separate models:
  - One forward pass per frame is required to keep up with line speed
    (typical machine-vision cameras run 15-60 fps; a 3-model pipeline would
    triple inference latency for no accuracy benefit, since the same visual
    features — edges, texture, color histograms — are useful for all three
    tasks).
  - Joint training with a shared trunk acts as a mild regularizer and
    generally improves the classification head's accuracy on small
    domain-specific datasets, which is the norm in manufacturing QC (you
    rarely have more than a few thousand labeled defect images per line).
"""

import torch
import torch.nn as nn
import torchvision.models as tvm

from constants import DEFECT_CLASSES  # re-exported for convenience


class DefectInspectionNet(nn.Module):
    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        num_classes: int = len(DEFECT_CLASSES),
        pretrained: bool = True,
        use_localization: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.use_localization = use_localization
        self.backbone_name = backbone

        if backbone == "efficientnet_b0":
            weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            net = tvm.efficientnet_b0(weights=weights)
            feat_dim = net.classifier[1].in_features
            net.classifier = nn.Identity()
            self.backbone = net

        elif backbone == "resnet50":
            weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            net = tvm.resnet50(weights=weights)
            feat_dim = net.fc.in_features
            net.fc = nn.Identity()
            self.backbone = net

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        hidden = 256
        self.trunk = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        # Head 1: defect classification
        self.classifier_head = nn.Linear(hidden, num_classes)

        # Head 2: severity regression (sigmoid-bounded to [0,1])
        self.severity_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Head 3 (optional): bounding box regression [x, y, w, h] in [0,1]
        if use_localization:
            self.bbox_head = nn.Sequential(
                nn.Linear(hidden, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 4),
                nn.Sigmoid(),
            )

    def unfreeze_backbone(self, last_n_blocks: int = None):
        """Call this for stage-2 fine-tuning after the heads have converged.
        If last_n_blocks is None, unfreezes the whole backbone; otherwise
        only the last N children (cheaper, less prone to catastrophic
        forgetting on small datasets)."""
        children = list(self.backbone.children())
        to_unfreeze = children if last_n_blocks is None else children[-last_n_blocks:]
        for block in to_unfreeze:
            for p in block.parameters():
                p.requires_grad = True

    def forward(self, x):
        feats = self.backbone(x)
        h = self.trunk(feats)
        out = {
            "defect_logits": self.classifier_head(h),
            "severity": self.severity_head(h).squeeze(-1),
        }
        if self.use_localization:
            out["bbox"] = self.bbox_head(h)
        return out


def build_model(config: dict) -> DefectInspectionNet:
    return DefectInspectionNet(
        backbone=config.get("backbone", "efficientnet_b0"),
        num_classes=len(config.get("classes", DEFECT_CLASSES)),
        pretrained=config.get("pretrained", True),
        use_localization=config.get("use_localization", True),
        freeze_backbone=config.get("freeze_backbone", True),
    )


if __name__ == "__main__":
    model = build_model({"backbone": "efficientnet_b0"})
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    for k, v in out.items():
        print(k, v.shape)
