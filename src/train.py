"""
train.py — Two-stage transfer learning for the defect inspection model.

Stage 1 (head warm-up): backbone frozen, only the classification/severity/
bbox heads train. Fast, prevents the randomly-initialized heads from
destroying the pretrained features with large early gradients.

Stage 2 (fine-tune): unfreeze the last few backbone blocks and train the
whole network end-to-end at a lower learning rate, so the ImageNet features
specialize to your material/lighting/part geometry.

Usage:
    python train.py --data_csv data/annotations.csv --image_dir data/images \
        --backbone efficientnet_b0 --epochs_stage1 8 --epochs_stage2 12

Loss = CE(defect_type) + lambda_sev * MSE(severity) + lambda_box * SmoothL1(bbox)
The severity and bbox terms are masked to defect-positive samples only
(an "ok" part has no meaningful severity or box).
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from model import build_model, DEFECT_CLASSES
from dataset import DefectDataset


def masked_mse(pred, target, mask):
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return ((pred[mask] - target[mask]) ** 2).mean()


def masked_smooth_l1(pred, target, mask):
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return nn.functional.smooth_l1_loss(pred[mask], target[mask])


def run_epoch(model, loader, device, optimizer=None, class_weights=None,
              lambda_sev=1.0, lambda_box=0.5):
    train_mode = optimizer is not None
    model.train(train_mode)

    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    total_loss, total_correct, n = 0.0, 0, 0

    ok_idx = DEFECT_CLASSES.index("ok")

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        severity = batch["severity"].to(device)
        bbox = batch["bbox"].to(device)
        defect_mask = labels != ok_idx

        with torch.set_grad_enabled(train_mode):
            out = model(images)
            loss = ce_loss_fn(out["defect_logits"], labels)
            loss = loss + lambda_sev * masked_mse(out["severity"], severity, defect_mask)
            if "bbox" in out:
                loss = loss + lambda_box * masked_smooth_l1(out["bbox"], bbox, defect_mask)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (out["defect_logits"].argmax(1) == labels).sum().item()
        n += images.size(0)

    return total_loss / n, total_correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--backbone", default="efficientnet_b0", choices=["efficientnet_b0", "resnet50"])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs_stage1", type=int, default=8)
    ap.add_argument("--epochs_stage2", type=int, default=12)
    ap.add_argument("--lr_stage1", type=float, default=1e-3)
    ap.add_argument("--lr_stage2", type=float, default=1e-5)
    ap.add_argument("--val_split", type=float, default=0.15)
    ap.add_argument("--out", default="qc_model.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_ds = DefectDataset(args.data_csv, args.image_dir, train=True)
    class_weights = full_ds.class_weights()

    val_len = int(len(full_ds) * args.val_split)
    train_ds, val_ds = random_split(full_ds, [len(full_ds) - val_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model({"backbone": args.backbone, "freeze_backbone": True}).to(device)

    # ---- Stage 1: warm up heads ----
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr_stage1
    )
    print(f"[Stage 1] training heads only, backbone frozen ({args.epochs_stage1} epochs)")
    for epoch in range(args.epochs_stage1):
        tr_loss, tr_acc = run_epoch(model, train_loader, device, optimizer, class_weights)
        val_loss, val_acc = run_epoch(model, val_loader, device, None, class_weights)
        print(f"  epoch {epoch+1}: train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    # ---- Stage 2: fine-tune last blocks ----
    model.unfreeze_backbone(last_n_blocks=2)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr_stage2
    )
    print(f"[Stage 2] fine-tuning last backbone blocks ({args.epochs_stage2} epochs)")
    best_val_acc = 0.0
    for epoch in range(args.epochs_stage2):
        tr_loss, tr_acc = run_epoch(model, train_loader, device, optimizer, class_weights)
        val_loss, val_acc = run_epoch(model, val_loader, device, None, class_weights)
        print(f"  epoch {epoch+1}: train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict(), "backbone": args.backbone}, args.out)
            print(f"    -> saved new best checkpoint to {args.out}")

    print(f"Done. Best val acc: {best_val_acc:.3f}")


if __name__ == "__main__":
    main()
