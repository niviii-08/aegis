"""WaveFake fine-tuning for AEGIS audio model.

Dataset: archive (1) — 10 vocoder directories (117,983 fake WAVs)
Real audio source: FoR-2sec real training split (6,978 WAVs)

Strategy:
- Real: FoR training/real  (human speech from 8 TTS-speakers, genuine)
- Fake: Balanced 700 samples from each of the 10 WaveFake vocoders (7,000 total)
- Then fine-tune from the best available checkpoint
- Saves to models/audio/wavefake_finetuned.pt

Design choices:
- 80/10/10 split of the assembled dataset (train/val/test)
- LR = 5e-6 (very conservative — avoids catastrophic forgetting)
- 6 epochs with early stopping (patience=3)
- Gradient clipping, OneCycleLR
- Optimal threshold calibration on val set
"""

from __future__ import annotations

import concurrent.futures
import logging
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

# ── Paths ──────────────────────────────────────────────────────────────────
WAVEFAKE_BASE = Path(r"C:\Users\Neevetha N\Downloads\archive (1)\generated_audio")
FOR_REAL_DIR = Path(
    r"C:\Users\Neevetha N\Downloads\archive\for-2sec\for-2seconds\training\real"
)
CACHE_DIR = Path("data/processed/audio/wavefake_cache")

# Prefer the FoR fine-tuned model; fall back to baseline
_CANDIDATES = [
    Path("models/audio/for_finetuned_full.pt"),
    Path("models/audio/for_finetuned.pt"),
    Path("models/audio/baseline_best.pt"),
]
CHECKPOINT_IN = next(p for p in _CANDIDATES if p.exists())
CHECKPOINT_OUT = Path("models/audio/wavefake_finetuned.pt")

# ── Hyperparameters ────────────────────────────────────────────────────────
SAMPLES_PER_VOCODER = 700   # 10 vocoders × 700 = 7,000 fake
TARGET_LENGTH = 600
BATCH_SIZE = 32
EPOCHS = 6
LR = 5e-6
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
MAX_WORKERS = 4
PATIENCE = 3
SEED = 42

VOCODER_DIRS = [
    "common_voices_prompts_from_conformer_fastspeech2_pwg_ljspeech",
    "jsut_multi_band_melgan",
    "jsut_parallel_wavegan",
    "ljspeech_full_band_melgan",
    "ljspeech_hifiGAN",
    "ljspeech_melgan",
    "ljspeech_melgan_large",
    "ljspeech_multi_band_melgan",
    "ljspeech_parallel_wavegan",
    "ljspeech_waveglow",
]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _extract_mel(path: Path) -> np.ndarray:
    """Extract 128-band log-mel spectrogram, resampled to 16 kHz."""
    y, sr = sf.read(str(path))
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
        mel = _extract_mel(src)
        np.save(str(dst), mel.astype(np.float32))
    except Exception as exc:
        logger.warning("Skip %s: %s", src.name, exc)


def _build_cache(
    tag: str,
    files: list[Path],
    label: int,
    prefix: str,
) -> list[tuple[Path, int]]:
    """Cache mel-specs and return (npy_path, label) pairs."""
    split_dir = CACHE_DIR / tag
    split_dir.mkdir(parents=True, exist_ok=True)

    samples: list[tuple[Path, int]] = []
    tasks: list[tuple[Path, Path]] = []
    for f in files:
        dst = split_dir / f"{prefix}_{f.stem}.npy"
        samples.append((dst, label))
        if not dst.exists():
            tasks.append((f, dst))

    if tasks:
        logger.info("  Extracting %d %s mel-specs...", len(tasks), prefix)
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            list(pool.map(_cache_worker, tasks))

    valid = [(p, lbl) for p, lbl in samples if p.exists()]
    return valid


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class AudioDataset(Dataset):
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
        t = torch.from_numpy(mel.astype(np.float32)).unsqueeze(0)
        if self.augment:
            t = self._spec_augment(t)
        return t, torch.tensor(float(label))

    def _fix_length(self, mel: np.ndarray) -> np.ndarray:
        _, T = mel.shape
        tgt = self.target_length
        if T > tgt:
            start = random.randint(0, T - tgt) if self.augment else (T - tgt) // 2
            mel = mel[:, start : start + tgt]
        elif T < tgt:
            pad = tgt - T
            mel = np.pad(mel, ((0, 0), (pad // 2, pad - pad // 2)), constant_values=0)
        return mel

    def _spec_augment(self, t: torch.Tensor) -> torch.Tensor:
        out = t.clone()
        _, n_mels, n_time = out.shape
        for _ in range(2):
            f = random.randint(1, 15)
            f0 = random.randint(0, max(0, n_mels - f))
            out[:, f0 : f0 + f, :] = 0.0
        for _ in range(2):
            tl = random.randint(1, 20)
            t0 = random.randint(0, max(0, n_time - tl))
            out[:, :, t0 : t0 + tl] = 0.0
        return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _forward(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for bx, by in loader:
            probs.extend(torch.sigmoid(model(bx.to(device))).cpu().tolist())
            labels.extend(by.numpy().astype(int).tolist())
    return np.array(labels), np.array(probs)


def _metrics(
    y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5
) -> dict[str, float]:
    y_pred = (y_prob >= thr).astype(int)
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


def _find_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cpu")
    logger.info("Device: %s | Checkpoint: %s", device, CHECKPOINT_IN)

    # ── 1. Collect source files ─────────────────────────────────────────────
    rng = random.Random(SEED)
    real_files = sorted(FOR_REAL_DIR.glob("*.wav"))
    rng.shuffle(real_files)

    fake_files: list[Path] = []
    for vdir in VOCODER_DIRS:
        wavs = sorted((WAVEFAKE_BASE / vdir).glob("*.wav"))
        rng.shuffle(wavs)
        picked = wavs[:SAMPLES_PER_VOCODER]
        fake_files.extend(picked)
        logger.info("  %s: %d samples", vdir, len(picked))

    rng.shuffle(fake_files)
    n_fake = len(fake_files)
    real_files = real_files[:n_fake]   # balance to fake count

    logger.info(
        "Dataset: %d real + %d fake = %d total",
        len(real_files), n_fake, len(real_files) + n_fake,
    )

    # ── 2. 80/10/10 split ──────────────────────────────────────────────────
    def split80(lst: list) -> tuple[list, list, list]:
        n = len(lst)
        t = int(0.8 * n)
        v = int(0.9 * n)
        return lst[:t], lst[t:v], lst[v:]

    tr_r, va_r, te_r = split80(real_files)
    tr_f, va_f, te_f = split80(fake_files)

    logger.info(
        "Split — Train: %d+%d | Val: %d+%d | Test: %d+%d",
        len(tr_r), len(tr_f), len(va_r), len(va_f), len(te_r), len(te_f),
    )

    # ── 3. Build cache ─────────────────────────────────────────────────────
    logger.info("=== Building WaveFake cache ===")
    train_s = (
        _build_cache("train", tr_r, 0, "real")
        + _build_cache("train", tr_f, 1, "fake")
    )
    val_s = (
        _build_cache("val", va_r, 0, "real")
        + _build_cache("val", va_f, 1, "fake")
    )
    test_s = (
        _build_cache("test", te_r, 0, "real")
        + _build_cache("test", te_f, 1, "fake")
    )

    logger.info(
        "Cache ready — Train: %d | Val: %d | Test: %d",
        len(train_s), len(val_s), len(test_s),
    )

    train_loader = DataLoader(
        AudioDataset(train_s, augment=True), batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        AudioDataset(val_s, augment=False), batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        AudioDataset(test_s, augment=False), batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0,
    )

    # ── 4. Load model ──────────────────────────────────────────────────────
    ckpt = torch.load(CHECKPOINT_IN, map_location=device, weights_only=False)
    model_cfg = ckpt.get("config", {}).get(
        "model",
        {"model_type": "mel_cnn", "n_mels": 128, "hidden_dim": 256,
         "num_conv_blocks": 4, "dropout": 0.3},
    )
    model = build_model(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info("Loaded %s ✓", CHECKPOINT_IN)

    y_true_pre, y_prob_pre = _forward(model, test_loader, device)
    pre = _metrics(y_true_pre, y_prob_pre)
    logger.info(
        "Pre-FT Test → Acc=%.1f%%  Prec=%.1f%%  Rec=%.1f%%  AUC=%.4f",
        pre["accuracy"]*100, pre["precision"]*100,
        pre["recall"]*100, pre["roc_auc"],
    )

    # ── 5. Fine-tune ───────────────────────────────────────────────────────
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR * 8,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS, pct_start=0.15, anneal_strategy="cos",
    )

    best_auc, best_weights, patience_count = 0.0, None, 0
    logger.info("Fine-tuning %d epochs @ LR=%.1e", EPOCHS, LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses: list[float] = []
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(bx), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

        y_true_v, y_prob_v = _forward(model, val_loader, device)
        val = _metrics(y_true_v, y_prob_v)
        logger.info(
            "Epoch %d/%d  Loss=%.4f  Val-Acc=%.1f%%  Val-AUC=%.4f  Val-F1=%.4f",
            epoch, EPOCHS, np.mean(losses),
            val["accuracy"]*100, val["roc_auc"], val["f1"],
        )

        if val["roc_auc"] > best_auc:
            best_auc = val["roc_auc"]
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
            logger.info("  ↳ New best val AUC: %.4f", best_auc)
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                logger.info("Early stop (patience=%d).", PATIENCE)
                break

    # ── 6. Threshold calibration + test eval ───────────────────────────────
    if best_weights:
        model.load_state_dict(best_weights)

    y_true_v, y_prob_v = _forward(model, val_loader, device)
    opt_t = _find_threshold(y_true_v, y_prob_v)
    logger.info("Optimal threshold (val balanced-acc): %.2f", opt_t)

    y_true_t, y_prob_t = _forward(model, test_loader, device)
    m_opt = _metrics(y_true_t, y_prob_t, opt_t)
    m05 = _metrics(y_true_t, y_prob_t, 0.5)

    logger.info(
        "\n=== WaveFake Test @ thr=0.50 ===\n"
        "  Acc=%.2f%%  Prec=%.2f%%  Rec=%.2f%%  F1=%.2f%%  AUC=%.4f",
        m05["accuracy"]*100, m05["precision"]*100,
        m05["recall"]*100, m05["f1"]*100, m05["roc_auc"],
    )
    logger.info(
        "\n=== WaveFake Test @ thr=%.2f (optimal) ===\n"
        "  Acc=%.2f%%  Prec=%.2f%%  Rec=%.2f%%  F1=%.2f%%  AUC=%.4f",
        opt_t,
        m_opt["accuracy"]*100, m_opt["precision"]*100,
        m_opt["recall"]*100, m_opt["f1"]*100, m_opt["roc_auc"],
    )

    # ── 7. Save checkpoint ─────────────────────────────────────────────────
    CHECKPOINT_OUT.parent.mkdir(parents=True, exist_ok=True)
    new_ckpt = dict(ckpt)
    new_ckpt["model_state_dict"] = model.state_dict()
    new_ckpt["wavefake_threshold"] = opt_t
    new_ckpt.setdefault("metrics", {})["wavefake_test"] = m_opt
    new_ckpt["epoch"] = ckpt.get("epoch", 5) + EPOCHS
    torch.save(new_ckpt, CHECKPOINT_OUT)
    logger.info("Saved → %s", CHECKPOINT_OUT)


if __name__ == "__main__":
    main()
