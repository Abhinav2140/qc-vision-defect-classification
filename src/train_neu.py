"""
train_neu.py — Transfer-learning defect classifier trained on the real NEU
Surface Defect dataset (6 classes of hot-rolled steel strip defects:
crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches).

This is the "real data, real numbers" counterpart to the synthetic-data demo
in generate_demo_data.py — it actually trains a CNN and reports genuine
accuracy/precision/recall/F1/confusion-matrix results, which is what the
notebook report in notebooks/defect_classification_report.ipynb is built
from (same logic, executed cell by cell with plots).
"""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)

DATA_DIR = Path(__file__).parent.parent / "data" / "neu"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# torchvision's default weight loader points at download.pytorch.org, which
# is unreachable from some sandboxed/restricted networks. timm's weights for
# these two backbones are hosted as GitHub release assets instead, which
# resolves in more restricted network environments — same ImageNet-trained
# weights, just a different, more reachable host.
TIMM_WEIGHT_URLS = {
    "resnet18": "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet18_a1_0-d63eafa0.pth",
    "efficientnet_b0": "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-tf-weights/efficientnet_b0_ra-3dd342df.pth",
}


def build_loaders(batch_size=64, image_size=96):
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.2),  # steel strip images have no fixed "up"
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(DATA_DIR / "val", transform=val_tf)

    # ImageFolder assigns class indices alphabetically — same order for both
    # splits since both dirs have identical subfolder names.
    assert train_ds.classes == val_ds.classes
    classes = train_ds.classes

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, classes


def build_model(num_classes, backbone="resnet18", freeze_backbone=True):
    if backbone not in ("resnet18", "efficientnet_b0"):
        raise ValueError(backbone)

    net = timm.create_model(
        backbone, pretrained=True, num_classes=num_classes,
        pretrained_cfg_overlay=dict(url=TIMM_WEIGHT_URLS[backbone], hf_hub_id=None),
    )

    if freeze_backbone:
        # Freeze everything, then re-enable the final classifier head
        # (timm names it "fc" for resnet18, "classifier" for efficientnet).
        for p in net.parameters():
            p.requires_grad = False
        head = net.get_classifier()
        for p in head.parameters():
            p.requires_grad = True

    return net.to(DEVICE)


def unfreeze_last_layer_block(model, backbone):
    if backbone == "resnet18":
        for p in model.layer4.parameters():
            p.requires_grad = True
    elif backbone == "efficientnet_b0":
        for p in model.blocks[-2:].parameters():
            p.requires_grad = True


def run_epoch(model, loader, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    loss_fn = nn.CrossEntropyLoss()
    total_loss, correct, n = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        with torch.set_grad_enabled(train_mode):
            logits = model(images)
            loss = loss_fn(logits, labels)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        n += images.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, classes):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(DEVICE)
        logits = model(images)
        preds = logits.argmax(1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(len(classes))), zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(classes))))
    report = classification_report(all_labels, all_preds, target_names=classes, zero_division=0)

    per_class = [
        {"class": c, "precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
        for c, p, r, f, s in zip(classes, precision, recall, f1, support)
    ]
    return {
        "accuracy": float(acc),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "labels": all_labels,
        "preds": all_preds,
    }


def main(backbone="resnet18", epochs_stage1=6, epochs_stage2=4, out_dir="../outputs"):
    out_dir = Path(__file__).parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, classes = build_loaders()
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")
    print(f"Device: {DEVICE}")

    model = build_model(len(classes), backbone=backbone, freeze_backbone=True)
    history = []

    print(f"\n[Stage 1] head-only training, backbone frozen ({epochs_stage1} epochs)")
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    for epoch in range(epochs_stage1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer)
        val_metrics = evaluate(model, val_loader, classes)
        history.append({"stage": 1, "epoch": epoch + 1, "train_loss": tr_loss,
                         "train_acc": tr_acc, "val_acc": val_metrics["accuracy"]})
        print(f"  epoch {epoch+1}: train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} "
              f"val_acc={val_metrics['accuracy']:.3f} ({time.time()-t0:.1f}s)")

    print(f"\n[Stage 2] fine-tuning last block ({epochs_stage2} epochs)")
    unfreeze_last_layer_block(model, backbone)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    for epoch in range(epochs_stage2):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer)
        val_metrics = evaluate(model, val_loader, classes)
        history.append({"stage": 2, "epoch": epoch + 1, "train_loss": tr_loss,
                         "train_acc": tr_acc, "val_acc": val_metrics["accuracy"]})
        print(f"  epoch {epoch+1}: train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} "
              f"val_acc={val_metrics['accuracy']:.3f} ({time.time()-t0:.1f}s)")

    final_metrics = evaluate(model, val_loader, classes)
    print("\n=== Final validation metrics ===")
    print(f"Accuracy: {final_metrics['accuracy']:.4f}")
    print(final_metrics["classification_report"])

    torch.save({"model_state": model.state_dict(), "backbone": backbone, "classes": classes},
               out_dir / "neu_defect_model.pt")
    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "final_metrics.json", "w") as f:
        json.dump({k: v for k, v in final_metrics.items() if k != "classification_report"}, f, indent=2)
    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(final_metrics["classification_report"])

    print(f"\nSaved model + metrics to {out_dir}")
    return model, history, final_metrics, classes


if __name__ == "__main__":
    main()
