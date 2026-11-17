"""Fine-tune AEGIS image model on the 140k Real vs StyleGAN Fake Faces dataset."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision.transforms as transforms
from image.models.factory import build_model


class ImageDataset(Dataset):
    """Dataset for loading facial image files with preprocessing."""

    def __init__(self, file_paths: list[Path], labels: list[int], transform: Any) -> None:
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        p = self.file_paths[idx]
        with Image.open(p) as img:
            img = img.convert("RGB")
            tensor = self.transform(img)
        return tensor, self.labels[idx]


def build_transform(input_size: int = 224) -> Any:
    """Build standard EfficientNet preprocessing transform."""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_dataset_splits(
    base_dir: Path,
    n_train: int = 1500,
    n_val: int = 500,
    n_test: int = 500,
) -> dict[str, tuple[list[Path], list[int]]]:
    """Collect balanced real and StyleGAN fake image paths for splits."""
    tr_real = sorted(list((base_dir / "train" / "real").glob("*.jpg")))[:n_train]
    tr_fake = sorted(list((base_dir / "train" / "fake").glob("*.jpg")))[:n_train]
    tr_paths = tr_real + tr_fake
    tr_labels = [0] * len(tr_real) + [1] * len(tr_fake)

    val_real = sorted(list((base_dir / "valid" / "real").glob("*.jpg")))[:n_val]
    val_fake = sorted(list((base_dir / "valid" / "fake").glob("*.jpg")))[:n_val]
    val_paths = val_real + val_fake
    val_labels = [0] * len(val_real) + [1] * len(val_fake)

    te_real = sorted(list((base_dir / "test" / "real").glob("*.jpg")))[:n_test]
    te_fake = sorted(list((base_dir / "test" / "fake").glob("*.jpg")))[:n_test]
    te_paths = te_real + te_fake
    te_labels = [0] * len(te_real) + [1] * len(te_fake)

    return {
        "train": (tr_paths, tr_labels),
        "val": (val_paths, val_labels),
        "test": (te_paths, te_labels),
    }


def extract_features(
    model: nn.Module,
    paths: list[Path],
    labels: list[int],
    batch_size: int = 32,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract 1792-dim features using the frozen EfficientNet backbone."""
    transform = build_transform()
    dataset = ImageDataset(paths, labels, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    features_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for i, (images, batch_labels) in enumerate(loader):
            images = images.to(device)
            feats = model.backbone(images)
            features_list.append(feats.cpu().numpy())
            labels_list.append(batch_labels.numpy())
            if (i + 1) % 15 == 0 or (i + 1) == len(loader):
                processed = min((i + 1) * batch_size, len(dataset))
                print(f"  Extracted {processed}/{len(dataset)} features...", flush=True)

    all_features = np.concatenate(features_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)
    return all_features, all_labels


def train_classifier_head(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    in_features: int = 1792,
    epochs: int = 8,
    device: str = "cpu",
) -> nn.Module:
    """Train the classifier head on extracted features."""
    head = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 1),
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    tr_dataset = TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float())
    tr_loader = DataLoader(tr_dataset, batch_size=64, shuffle=True)

    best_val_auc = 0.0
    best_weights: dict[str, Any] = {}

    for epoch in range(1, epochs + 1):
        head.train()
        total_loss = 0.0
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = head(bx).squeeze(-1)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(by)
        scheduler.step()

        val_auc, val_acc = evaluate_head(head, val_x, val_y, device)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_weights = {k: v.cpu().clone() for k, v in head.state_dict().items()}

        print(f"  Epoch {epoch:2d}/{epochs} | Loss: {total_loss/len(train_y):.4f} | Val Acc: {val_acc*100:5.2f}% | Val AUC: {val_auc:6.4f}", flush=True)

    head.load_state_dict(best_weights)
    return head


def evaluate_head(
    head: nn.Module, x: np.ndarray, y: np.ndarray, device: str = "cpu"
) -> tuple[float, float]:
    """Compute ROC-AUC and accuracy for a feature set."""
    head.eval()
    with torch.no_grad():
        logits = head(torch.from_numpy(x).float().to(device)).squeeze(-1).cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= 0.5).astype(int)
    auc = float(roc_auc_score(y, probs))
    acc = float(accuracy_score(y, preds))
    return auc, acc


def evaluate_test_set(
    head: nn.Module, test_x: np.ndarray, test_y: np.ndarray, device: str = "cpu"
) -> dict[str, Any]:
    """Compute comprehensive test metrics on holdout test set."""
    head.eval()
    with torch.no_grad():
        logits = head(torch.from_numpy(test_x).float().to(device)).squeeze(-1).cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(test_y, preds)
    prec = precision_score(test_y, preds, zero_division=0)
    rec = recall_score(test_y, preds, zero_division=0)
    f1 = f1_score(test_y, preds, zero_division=0)
    auc = roc_auc_score(test_y, probs)
    cm = confusion_matrix(test_y, preds).tolist()

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": cm,
        "sample_count": len(test_y),
    }


def save_finetuned_checkpoint(
    base_ckpt_path: Path,
    head: nn.Module,
    test_metrics: dict[str, Any],
    out_path: Path,
    epochs: int = 8,
) -> None:
    """Save full model checkpoint matching AEGIS factory format."""
    base_ckpt = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
    state_dict = base_ckpt["model_state_dict"]

    head_state = head.state_dict()
    state_dict["classifier.1.weight"] = head_state["1.weight"].cpu()
    state_dict["classifier.1.bias"] = head_state["1.bias"].cpu()

    out_ckpt = {
        "epoch": epochs,
        "model_state_dict": state_dict,
        "model_architecture": base_ckpt.get("model_architecture", {}),
        "config": base_ckpt.get("config", {}),
        "metrics": {"test": test_metrics},
        "dataset": "140k Real vs StyleGAN Fake Faces",
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, out_path)
    print(f"\nSaved fine-tuned StyleGAN image checkpoint to: {out_path}", flush=True)


def run_sample_verification(ckpt_path: Path, test_dir: Path) -> None:
    """Run live inference on 10 Real and 10 AI fake images."""
    from api.services.image_service import ImageInferenceService

    svc = ImageInferenceService(ckpt_path, device="cpu")
    reals = sorted(list((test_dir / "real").glob("*.jpg")))[:10]
    fakes = sorted(list((test_dir / "fake").glob("*.jpg")))[:10]

    print("\n" + "=" * 80)
    print("  LIVE VERIFICATION: 10 REAL VS 10 AI SAMPLES (STYLEGAN MODEL)")
    print("=" * 80)
    print("REAL SAMPLES (FFHQ):")
    real_pass = 0
    for r in reals:
        res = svc.predict(r)
        pred = res["prediction"].upper()
        prob = res["probability"]
        status = "[PASS]" if pred == "REAL" else "[FAIL]"
        if pred == "REAL":
            real_pass += 1
        print(f"  {r.name:<18} | Pred: {pred:<4} (P(Fake)={prob:.4f}) | {status}")

    print("\nAI FAKE SAMPLES (STYLEGAN):")
    fake_pass = 0
    for f in fakes:
        res = svc.predict(f)
        pred = res["prediction"].upper()
        prob = res["probability"]
        status = "[PASS]" if pred == "FAKE" else "[FAIL]"
        if pred == "FAKE":
            fake_pass += 1
        print(f"  {f.name:<18} | Pred: {pred:<4} (P(Fake)={prob:.4f}) | {status}")

    print(f"\nSummary: Real {real_pass}/10 | Fake {fake_pass}/10 | Overall: {(real_pass + fake_pass)}/20")


def main() -> None:
    """Execute complete 140k Real vs StyleGAN fine-tuning workflow."""
    base_dir = Path("data/raw/image/real_vs_fake/real-vs-fake")
    base_ckpt_path = Path("models/image/baseline_best.pt")
    out_ckpt_path = Path("models/image/stylegan_finetuned.pt")
    cache_dir = Path("data/processed/image/stylegan_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  AEGIS IMAGE MODEL FINE-TUNING (140K REAL VS STYLEGAN FAKE FACES)")
    print("=" * 80)

    # 1. Load sample paths
    splits = load_dataset_splits(base_dir, n_train=1000, n_val=300, n_test=500)
    for s_name, (p_list, l_list) in splits.items():
        print(f"  Split [{s_name:<5}]: {len(p_list)} images (Real: {l_list.count(0)}, Fake: {l_list.count(1)})")

    # 2. Load base model
    print(f"\nLoading baseline model: {base_ckpt_path}")
    base_ckpt = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
    model_cfg = dict(base_ckpt.get("config", {}).get("model", {}))
    model_cfg["pretrained"] = False
    model = build_model(model_cfg)
    model.load_state_dict(base_ckpt["model_state_dict"])
    model.eval()

    # 3. Extract features (with disk cache)
    features: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for s_name, (paths, labels) in splits.items():
        feat_path = cache_dir / f"{s_name}_feats.npy"
        lbl_path = cache_dir / f"{s_name}_lbls.npy"
        if feat_path.exists() and lbl_path.exists():
            print(f"Loading cached {s_name} features from disk...")
            features[s_name] = (np.load(feat_path), np.load(lbl_path))
        else:
            print(f"\nExtracting {s_name} features ({len(paths)} images)...")
            t0 = time.time()
            feats, lbls = extract_features(model, paths, labels)
            print(f"  Done in {time.time()-t0:.1f}s. Shape: {feats.shape}")
            np.save(feat_path, feats)
            np.save(lbl_path, lbls)
            features[s_name] = (feats, lbls)

    # 4. Train classifier head for 8 epochs
    print("\nTraining classifier head on StyleGAN features (8 Epochs)...")
    tr_x, tr_y = features["train"]
    val_x, val_y = features["val"]
    head = train_classifier_head(tr_x, tr_y, val_x, val_y, epochs=8)

    # 5. Evaluate on holdout test set
    print("\nEvaluating on holdout test set (1,000 images)...")
    te_x, te_y = features["test"]
    metrics = evaluate_test_set(head, te_x, te_y)
    print("-" * 80)
    print(f"  Test Accuracy  : {metrics['accuracy']*100:.2f}%")
    print(f"  Test Precision : {metrics['precision']*100:.2f}%")
    print(f"  Test Recall    : {metrics['recall']*100:.2f}%")
    print(f"  Test F1-Score  : {metrics['f1']*100:.2f}%")
    print(f"  Test ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"  Confusion Mat  : {metrics['confusion_matrix']} [TN, FP / FN, TP]")
    print("-" * 80)

    # 6. Save fine-tuned checkpoint
    save_finetuned_checkpoint(base_ckpt_path, head, metrics, out_ckpt_path, epochs=8)

    # 7. Run sample verification on 10 real and 10 AI samples
    run_sample_verification(out_ckpt_path, base_dir / "test")


if __name__ == "__main__":
    main()
