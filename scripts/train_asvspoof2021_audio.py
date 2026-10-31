"""Fine-tune AEGIS audio model on ASVspoof 2021 dataset (archive 2).

Extracts mel-spectrograms from ASVspoof 2021 LA and DF evaluation sets,
covering lossy compression codecs (mp3, m4a, opus, alaw, ulaw, g722)
and advanced TTS/VC synthesis attacks.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import random
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

# Add src to path
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audio.models.factory import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Paths
ARCHIVE2_BASE = Path(r"C:\Users\Neevetha N\Downloads\archive (2)")
CACHE_DIR = Path("data/processed/audio/asvspoof2021_cache")
CHECKPOINT_IN = Path("models/audio/for_finetuned_full.pt")
CHECKPOINT_OUT = Path("models/audio/asvspoof2021_finetuned.pt")

# Hyperparameters
TARGET_LENGTH = 600
BATCH_SIZE = 32
EPOCHS = 5
LR = 8e-6
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
MAX_WORKERS = 6
SEED = 42


def extract_mel_db(file_path: Path) -> np.ndarray:
    """Read a 16kHz audio file and extract 128-band dB mel-spectrogram."""
    y, sr = sf.read(str(file_path))
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != 16000:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=16000)
        sr = 16000
    mel = librosa.feature.melspectrogram(
        y=y.astype(np.float32), sr=sr, n_fft=512, hop_length=160, n_mels=128
    )
    return librosa.power_to_db(mel, ref=np.max)


def _cache_worker(args: tuple[Path, Path]) -> None:
    src, dst = args
    try:
        mel = extract_mel_db(src)
        np.save(str(dst), mel.astype(np.float32))
    except Exception as exc:
        logger.warning("Failed to process %s: %s", src.name, exc)


def build_split_cache(
    split_name: str,
    real_files: list[Path],
    fake_files: list[Path],
) -> list[tuple[Path, int]]:
    """Extract and cache mel-spectrograms for a split."""
    split_dir = CACHE_DIR / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    samples: list[tuple[Path, int]] = []
    tasks: list[tuple[Path, Path]] = []

    for f in real_files:
        dst = split_dir / f"real_{f.stem}.npy"
        samples.append((dst, 0))
        if not dst.exists():
            tasks.append((f, dst))

    for f in fake_files:
        dst = split_dir / f"fake_{f.stem}.npy"
        samples.append((dst, 1))
        if not dst.exists():
            tasks.append((f, dst))

    if tasks:
        logger.info("Extracting %d mel-specs for '%s'...", len(tasks), split_name)
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            list(pool.map(_cache_worker, tasks))

    valid = [(p, lbl) for p, lbl in samples if p.exists()]
    logger.info("Split '%s': %d valid cached samples", split_name, len(valid))
    return valid


class ASVspoof2021Dataset(Dataset):
    """Dataset for ASVspoof 2021 cached mel-spectrograms."""

    def __init__(
        self,
        samples: list[tuple[Path, int]],
        target_length: int = TARGET_LENGTH,
        augment: bool = False,
    ) -> None:
        self.samples = samples
        self.target_length = target_length
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        npy_path, label = self.samples[idx]
        mel: np.ndarray = np.load(str(npy_path))
        if mel.ndim == 3 and mel.shape[0] == 1:
            mel = mel[0]

        mel = self._fix_length(mel)
        tensor = torch.from_numpy(mel.astype(np.float32)).unsqueeze(0)
        if self.augment:
            tensor = self._spec_augment(tensor)
        return tensor, torch.tensor(float(label))

    def _fix_length(self, mel: np.ndarray) -> np.ndarray:
        _, t_steps = mel.shape
        if t_steps > self.target_length:
            start = (
                random.randint(0, t_steps - self.target_length)
                if self.augment
                else (t_steps - self.target_length) // 2
            )
            mel = mel[:, start : start + self.target_length]
        elif t_steps < self.target_length:
            pad = self.target_length - t_steps
            mel = np.pad(mel, ((0, 0), (pad // 2, pad - pad // 2)), constant_values=0)
        return mel

    def _spec_augment(self, tensor: torch.Tensor) -> torch.Tensor:
        out = tensor.clone()
        _, n_mels, t_len = out.shape
        for _ in range(2):
            f = random.randint(1, 15)
            f0 = random.randint(0, max(0, n_mels - f))
            out[:, f0 : f0 + f, :] = 0.0
        for _ in range(2):
            tl = random.randint(1, 20)
            t0 = random.randint(0, max(0, t_len - tl))
            out[:, :, t0 : t0 + tl] = 0.0
        return out


def evaluate_model(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Forward pass to gather labels and predicted probabilities."""
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for bx, by in loader:
            probs.extend(torch.sigmoid(model(bx.to(device))).cpu().tolist())
            labels.extend(by.numpy().astype(int).tolist())
    return np.array(labels), np.array(probs)


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Calculate classification metrics at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(y_true, y_prob))
            if len(np.unique(y_true)) > 1
            else 0.5
        ),
    }


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find threshold that maximizes balanced accuracy on validation set."""
    best_t, best_ba = 0.5, 0.0
    for t in np.arange(0.20, 0.81, 0.01):
        y_pred = (y_prob >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        ba = (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)) / 2
        if ba > best_ba:
            best_ba, best_t = ba, t
    return float(best_t)


def _scan_meta(
    meta_path: Path, dirs: list[Path], target_count: int, rng: random.Random
) -> tuple[list[Path], list[Path]]:
    """Scan metadata lines and find working flac files up to target_count."""
    real_files, fake_files = [], []
    if not meta_path.exists():
        return real_files, fake_files

    with open(meta_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    rng.shuffle(lines)

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        fid, label = parts[1], parts[5]
        target_list = real_files if label == "bonafide" else fake_files
        if len(target_list) >= target_count:
            continue

        for d in dirs:
            fp = d / f"{fid}.flac"
            if fp.exists():
                try:
                    with sf.SoundFile(str(fp)) as sfile:
                        if sfile.frames > 0:
                            target_list.append(fp)
                except Exception:
                    pass
                break

        if len(real_files) >= target_count and len(fake_files) >= target_count:
            break

    return real_files, fake_files


def collect_asvspoof2021_files() -> tuple[list[Path], list[Path]]:
    """Parse metadata and collect decodable real and fake audio files."""
    rng = random.Random(SEED)
    la_meta = ARCHIVE2_BASE / "LA-keys-full" / "keys" / "LA" / "CM" / "trial_metadata.txt"
    la_dir = ARCHIVE2_BASE / "ASVspoof2021_LA_eval" / "ASVspoof2021_LA_eval" / "flac"

    df_meta = ARCHIVE2_BASE / "DF-keys-full" / "keys" / "DF" / "CM" / "trial_metadata.txt"
    df_dirs = [
        ARCHIVE2_BASE / f"ASVspoof2021_DF_eval_part0{i}" / "ASVspoof2021_DF_eval" / "flac"
        for i in range(3)
    ]

    logger.info("Scanning ASVspoof 2021 candidates...")
    la_real, la_fake = _scan_meta(la_meta, [la_dir], target_count=1500, rng=rng)
    df_real, df_fake = _scan_meta(df_meta, df_dirs, target_count=1500, rng=rng)

    real_all = la_real + df_real
    fake_all = la_fake + df_fake
    rng.shuffle(real_all)
    rng.shuffle(fake_all)

    min_count = min(len(real_all), len(fake_all))
    real_all, fake_all = real_all[:min_count], fake_all[:min_count]
    logger.info("Collected %d real + %d fake files from ASVspoof 2021", len(real_all), len(fake_all))
    return real_all, fake_all


def create_dataloaders(
    real_files: list[Path], fake_files: list[Path]
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Prepare train, val, test datasets and loaders."""
    total = len(real_files)
    n_tr, n_va = int(total * 0.66), int(total * 0.17)

    tr_r, va_r, te_r = real_files[:n_tr], real_files[n_tr : n_tr + n_va], real_files[n_tr + n_va :]
    tr_f, va_f, te_f = fake_files[:n_tr], fake_files[n_tr : n_tr + n_va], fake_files[n_tr + n_va :]

    train_s = build_split_cache("train", tr_r, tr_f)
    val_s = build_split_cache("val", va_r, va_f)
    test_s = build_split_cache("test", te_r, te_f)

    train_loader = DataLoader(
        ASVspoof2021Dataset(train_s, augment=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        ASVspoof2021Dataset(val_s, augment=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        ASVspoof2021Dataset(test_s, augment=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader, test_loader


def run_training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> dict:
    """Execute fine-tuning epochs and return best model state dict."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR * 5,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=0.2,
        anneal_strategy="cos",
    )

    best_auc, best_weights = 0.0, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(bx), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

        y_true_v, y_prob_v = evaluate_model(model, val_loader, device)
        val = compute_metrics(y_true_v, y_prob_v)
        logger.info(
            "Epoch %d/%d Loss=%.4f Val-Acc=%.1f%% Val-AUC=%.4f Val-F1=%.4f",
            epoch, EPOCHS, np.mean(losses), val["accuracy"] * 100, val["roc_auc"], val["f1"]
        )

        if val["roc_auc"] > best_auc:
            best_auc = val["roc_auc"]
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info("  ↳ New best val AUC: %.4f", best_auc)

    return best_weights or {k: v.cpu().clone() for k, v in model.state_dict().items()}


def save_and_report(
    model: nn.Module,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    ckpt: dict,
) -> None:
    """Calibrate optimal threshold, evaluate test split, and save checkpoint."""
    y_true_v, y_prob_v = evaluate_model(model, val_loader, device)
    opt_thr = find_optimal_threshold(y_true_v, y_prob_v)
    logger.info("Calibrated optimal threshold: %.2f", opt_thr)

    y_true_t, y_prob_t = evaluate_model(model, test_loader, device)
    m05 = compute_metrics(y_true_t, y_prob_t, 0.5)
    m_opt = compute_metrics(y_true_t, y_prob_t, opt_thr)

    logger.info(
        "\n=== ASVspoof 2021 Test @ 0.50 ===\n"
        "  Acc=%.2f%%  Prec=%.2f%%  Rec=%.2f%%  F1=%.2f%%  AUC=%.4f",
        m05["accuracy"] * 100, m05["precision"] * 100, m05["recall"] * 100, m05["f1"] * 100, m05["roc_auc"]
    )
    logger.info(
        "\n=== ASVspoof 2021 Test @ %.2f (optimal) ===\n"
        "  Acc=%.2f%%  Prec=%.2f%%  Rec=%.2f%%  F1=%.2f%%  AUC=%.4f",
        opt_thr, m_opt["accuracy"] * 100, m_opt["precision"] * 100, m_opt["recall"] * 100, m_opt["f1"] * 100, m_opt["roc_auc"]
    )

    CHECKPOINT_OUT.parent.mkdir(parents=True, exist_ok=True)
    new_ckpt = dict(ckpt)
    new_ckpt["model_state_dict"] = model.state_dict()
    new_ckpt["asvspoof2021_threshold"] = opt_thr
    new_ckpt.setdefault("metrics", {})["asvspoof2021_test"] = m_opt
    new_ckpt["epoch"] = ckpt.get("epoch", 17) + EPOCHS
    torch.save(new_ckpt, CHECKPOINT_OUT)
    logger.info("Saved ASVspoof 2021 fine-tuned model → %s", CHECKPOINT_OUT)


def main() -> None:
    """Main training orchestrator."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cpu")
    logger.info("Using device: %s", device)

    real_files, fake_files = collect_asvspoof2021_files()
    train_loader, val_loader, test_loader = create_dataloaders(real_files, fake_files)

    ckpt = torch.load(CHECKPOINT_IN, map_location=device, weights_only=False)
    model_cfg = ckpt.get("config", {}).get(
        "model",
        {"model_type": "mel_cnn", "n_mels": 128, "hidden_dim": 256, "num_conv_blocks": 4, "dropout": 0.3},
    )
    model = build_model(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info("Loaded checkpoint %s ✓", CHECKPOINT_IN)

    y_true_pre, y_prob_pre = evaluate_model(model, test_loader, device)
    pre = compute_metrics(y_true_pre, y_prob_pre)
    logger.info("Pre-FT ASVspoof 2021 Test: Acc=%.2f%%, AUC=%.4f", pre["accuracy"] * 100, pre["roc_auc"])

    best_weights = run_training_loop(model, train_loader, val_loader, device)
    model.load_state_dict(best_weights)
    save_and_report(model, val_loader, test_loader, device, ckpt)


if __name__ == "__main__":
    main()
