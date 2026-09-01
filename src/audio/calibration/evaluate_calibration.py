"""AEGIS Audio Calibration Evaluation — ECE, reliability diagrams, Brier, NLL, AUC, F1.

Calibration fitness is assessed *per split*:

  * validation   — the split on which temperature is fitted
  * test_seen    — seen generators; must not be used for fitting
  * test_unseen  — novel generators; must not be used for fitting

The script:

1. Collects raw logits from the (calibrated or base) model on each split.
2. If the checkpoint carries a ``temperature`` key, applies it automatically.
3. Also runs the base model (T=1) for a before/after comparison.
4. Computes before/after calibration metrics on ALL splits.
5. Produces reliability diagrams (before and after) saved to
   ``reports/audio/calibration/``.
6. Writes calibration_report.md and calibration_result.json.

Usage::

    python -m audio.calibration.evaluate_calibration \\
        --checkpoint  models/audio/baseline_calibrated.pt \\
        --config      configs/audio_baseline.yaml \\
        --output-dir  reports/audio/calibration

Terminology note
----------------
The calibration record stores ``probability`` = sigmoid(logit / T).
This is NEVER called a "confidence interval" — that term is reserved for
statistical interval estimation.  A future conformal prediction module will
add coverage-guaranteed prediction sets as a SEPARATE output key.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── bootstrap src/ on sys.path ────────────────────────────────────────────────
_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from audio.models.factory import build_model
from audio.training.dataset import AudioDataset, resolve_samples
from audio.training.utils import (
    find_project_root,
    load_training_config,
    resolve_device,
    set_seed,
    should_use_amp,
    get_git_commit,
)
from audio.calibration.temperature_scaling import (
    CalibrationRecord,
    fit_temperature,
    save_calibration,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Calibration metrics  (identical implementation to image counterpart)
# ─────────────────────────────────────────────────────────────────────────────

def _safe(fn, *args, default=None, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return default


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Expected Calibration Error + bin data for reliability diagram.

    Returns
    -------
    ece : float
    bin_accs : (n_bins,) mean accuracy per bin
    bin_confs : (n_bins,) mean confidence per bin
    bin_counts : (n_bins,) sample count per bin
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        if mask.sum() == 0:
            continue
        bin_accs[i] = y_true[mask].mean()
        bin_confs[i] = y_prob[mask].mean()
        bin_counts[i] = int(mask.sum())

    n = len(y_true)
    ece = float(np.sum(bin_counts * np.abs(bin_accs - bin_confs)) / max(n, 1))
    return ece, bin_accs, bin_confs, bin_counts


def compute_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def compute_nll(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    eps = 1e-7
    p = np.clip(y_prob, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def compute_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, y_prob))


def compute_f1(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    from sklearn.metrics import f1_score
    y_pred = (y_prob >= threshold).astype(int)
    return float(f1_score(y_true, y_pred, zero_division=0))


def full_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """All calibration + classification metrics for one split."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    if y_true.size == 0:
        return {"support": 0, "note": "empty_split"}

    ece, bin_accs, bin_confs, bin_counts = compute_ece(y_true, y_prob, n_bins=n_bins)
    return {
        "support": int(y_true.size),
        "ece": round(ece, 6),
        "brier_score": round(compute_brier(y_true, y_prob), 6),
        "nll": round(compute_nll(y_true, y_prob), 6),
        "roc_auc": _safe(compute_roc_auc, y_true, y_prob),
        "f1": round(compute_f1(y_true, y_prob, threshold=threshold), 6),
        "reliability_diagram": {
            "bin_accs": bin_accs.tolist(),
            "bin_confs": bin_confs.tolist(),
            "bin_counts": bin_counts.tolist(),
            "n_bins": n_bins,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Logit collection (audio-specific — no sample_ids in the loader)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (logits, labels) arrays from a DataLoader."""
    model.eval()
    all_logits: list[float] = []
    all_labels: list[int] = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(batch_x)
        all_logits.extend(logits.cpu().float().numpy().tolist())
        all_labels.extend(batch_y.numpy().astype(int).tolist())

    return np.array(all_logits, dtype=np.float64), np.array(all_labels, dtype=np.float64)


def load_checkpoint_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load audio model from checkpoint, rebuilding from stored config."""
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_dict = payload.get("config", {})
    model_cfg = config_dict.get("model", {})
    model = build_model(model_cfg)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    return model, payload


def build_loader(
    split_csv: Path,
    config: Any,
    device: torch.device,
) -> DataLoader | None:
    """Build an audio DataLoader for one split, or return None if unavailable."""
    if not split_csv.is_file():
        logger.warning("Split CSV not found: %s", split_csv)
        return None
    samples = resolve_samples(
        split_csv,
        config.metadata_path,
        config.project_root,
        preprocessing_version=config.preprocessing_version,
        feature_mode=config.feature_mode,
    )
    if not samples:
        logger.warning("No preprocessed samples for %s", split_csv.stem)
        return None
    return DataLoader(
        AudioDataset(samples, feature_mode=config.feature_mode, is_training=False),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and device.type == "cuda",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Reliability diagram
# ─────────────────────────────────────────────────────────────────────────────

def _draw_reliability_diagram(
    bin_confs: np.ndarray,
    bin_accs: np.ndarray,
    bin_counts: np.ndarray,
    ece: float,
    title: str,
    output_path: Path,
) -> None:
    """Render a reliability diagram to a PNG file."""
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend
    import matplotlib.pyplot as plt

    n_bins = len(bin_confs)
    centres = np.linspace(0, 1, n_bins, endpoint=False) + 0.5 / n_bins
    width = 1.0 / n_bins

    fig, ax = plt.subplots(figsize=(6, 6))

    # Gap bars (over/underconfidence shading)
    for i in range(n_bins):
        if bin_counts[i] == 0:
            continue
        conf = bin_confs[i]
        acc = bin_accs[i]
        # Perfect bar (grey outline)
        ax.bar(centres[i], conf, width=width * 0.9, color="lightgrey", edgecolor="grey",
               linewidth=0.8, zorder=1)
        # Actual accuracy bar
        color = "#2196F3" if acc >= conf else "#F44336"
        ax.bar(centres[i], acc, width=width * 0.9, color=color, alpha=0.8, zorder=2)

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect calibration")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability (confidence)", fontsize=11)
    ax.set_ylabel("Fraction of positives (accuracy)", fontsize=11)
    ax.set_title(f"{title}\nECE = {ece:.4f}", fontsize=12)
    ax.legend(fontsize=9)

    # Sample-count annotation
    for i in range(n_bins):
        if bin_counts[i] > 0:
            ax.text(centres[i], 0.02, str(bin_counts[i]),
                    ha="center", fontsize=6, color="black", zorder=3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Reliability diagram → %s", output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Markdown report  (mirrors image counterpart, audio-specific labels)
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(v: Any, d: int = 4) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{d}f}"
    return str(v)


def _metric_row(split: str, m: dict[str, Any], label: str = "") -> str:
    return (
        f"| {label or split} | {_fmt(m.get('ece'))} | {_fmt(m.get('brier_score'))} "
        f"| {_fmt(m.get('nll'))} | {_fmt(m.get('roc_auc'))} | {_fmt(m.get('f1'))} "
        f"| {m.get('support', 0)} |"
    )


def build_markdown_report(
    result: dict[str, Any],
    output_dir: Path,
) -> str:
    before = result["before_calibration"]
    after = result["after_calibration"]
    cal = result["calibration_record"]
    splits = ("val", "test_seen", "test_unseen")

    lines = [
        "# AEGIS Audio Confidence Calibration Report",
        "",
        f"> Generated: `{result['generated_at']}`  ",
        f"> Checkpoint: `{result['checkpoint_path']}`  ",
        f"> Git commit: `{result.get('git_commit', 'N/A')}`",
        "",
        "---",
        "",
        "## Terminology",
        "",
        "| Term | Definition |",
        "|------|------------|",
        "| **probability** | Calibrated P(fake) = sigmoid(logit / T). A number in [0,1]. |",
        "| **temperature** | Scalar T learned from the validation set to make probabilities reliable. |",
        "| **calibrated** | Whether temperature scaling has been applied (boolean). |",
        "| **confidence** | *Not yet implemented.* Reserved for a future model-reliability score |",
        "|  | (e.g. ensemble agreement, distance to training distribution). |",
        "| **conformal prediction** | *Not yet implemented.* When added, will produce prediction sets |",
        "|  | with a guaranteed coverage rate. Stored as a *separate key*, never conflated with probability. |",
        "",
        "> Do not call `probability` a 'confidence interval'. These are distinct concepts.",
        "",
        "---",
        "",
        "## Temperature Scaling",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Fitted temperature T | `{cal['temperature']:.6f}` |",
        f"| Fitted on split | `{cal['fitted_on_split']}` (validation only — never test) |",
        f"| NLL before | `{cal['fit_info'].get('nll_before_calibration', 'N/A')}` |",
        f"| NLL after | `{cal['fit_info'].get('nll_after_calibration', 'N/A')}` |",
        f"| NLL reduction | `{cal['fit_info'].get('nll_reduction', 'N/A')}` |",
        f"| Converged | `{cal['fit_info'].get('converged', 'N/A')}` |",
        f"| n calibration samples | `{cal['fit_info'].get('n_calibration_samples', 'N/A')}` |",
        "",
        "> T > 1 → model was **overconfident** (outputs flattened toward 0.5).  ",
        "> T < 1 → model was **underconfident** (outputs sharpened).  ",
        "> T = 1 → no correction needed.  ",
        "",
        "---",
        "",
        "## Per-Split Metrics",
        "",
        "### Before Calibration",
        "",
        "| Split | ECE ↓ | Brier ↓ | NLL ↓ | ROC-AUC ↑ | F1 ↑ | Support |",
        "|-------|--------|---------|-------|-----------|------|---------|",
    ]

    for sp in splits:
        m = before.get(sp, {})
        if m.get("support", 0) == 0:
            lines.append(f"| {sp} | — | — | — | — | — | 0 (empty) |")
        else:
            lines.append(_metric_row(sp, m))

    lines += [
        "",
        "### After Calibration",
        "",
        "| Split | ECE ↓ | Brier ↓ | NLL ↓ | ROC-AUC ↑ | F1 ↑ | Support |",
        "|-------|--------|---------|-------|-----------|------|---------|",
    ]
    for sp in splits:
        m = after.get(sp, {})
        if m.get("support", 0) == 0:
            lines.append(f"| {sp} | — | — | — | — | — | 0 (empty) |")
        else:
            lines.append(_metric_row(sp, m))

    lines += [
        "",
        "### Delta (After − Before)",
        "",
        "Negative Δ for ECE/Brier/NLL means **improvement**.",
        "",
        "| Split | ΔECE | ΔBrier | ΔNLL | ΔROC-AUC | ΔF1 |",
        "|-------|------|--------|------|----------|-----|",
    ]
    for sp in splits:
        b = before.get(sp, {})
        a = after.get(sp, {})
        if b.get("support", 0) == 0:
            lines.append(f"| {sp} | — | — | — | — | — |")
            continue

        def delta(key: str) -> str:
            bv = b.get(key)
            av = a.get(key)
            if bv is None or av is None:
                return "N/A"
            d = av - bv
            return f"{d:+.4f}"

        lines.append(
            f"| {sp} | {delta('ece')} | {delta('brier_score')} | {delta('nll')} "
            f"| {delta('roc_auc')} | {delta('f1')} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Reliability Diagrams",
        "",
        "Each bar represents a probability bin. Blue = model underestimates; Red = model overestimates.",
        "Sample counts are printed at the base of each bar.",
        "",
        "### Before Calibration",
        "",
        "![Reliability diagram before calibration](reliability_before.png)",
        "",
        "### After Calibration",
        "",
        "![Reliability diagram after calibration](reliability_after.png)",
        "",
        "---",
        "",
        "## Prediction API Example",
        "",
        "```json",
        json.dumps({
            "label": "fake",
            "probability": 0.73,
            "calibrated": True,
            "temperature": round(cal["temperature"], 4),
            "note": (
                "probability is calibrated P(fake) via temperature scaling. "
                "This is not a confidence interval. "
                "Conformal prediction sets (coverage-guaranteed) are not yet implemented "
                "and will be represented as a separate key when added."
            ),
        }, indent=2),
        "```",
        "",
        "---",
        "",
        "## Limitations",
        "",
        "- **Single-parameter calibration**: Temperature scaling cannot correct "
          "systematic bias per class or per sub-group. If the model is better "
          "calibrated on one generator than another, per-generator calibration "
          "or isotonic regression may be needed.",
        "- **test_unseen is empty**: Calibration quality on novel generators is "
          "unknown until unseen samples are preprocessed.",
        "- **Small calibration set**: A small validation set makes the fitted T "
          "noisy. ECE estimates with few samples per bin should be treated with caution.",
        "",
        "---",
        "*AEGIS Audio Calibration — automated, reproducible.*",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_calibration_evaluation(
    checkpoint_path: Path,
    config_path: Path,
    output_dir: Path,
    *,
    project_root: Path | None = None,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Full audio calibration evaluation pipeline.

    Steps
    -----
    1. Load model + config.
    2. Collect (logits, labels) for val / test_seen / test_unseen.
    3. Read temperature T from checkpoint (if present), or fit on val only.
    4. Compute before/after metrics on all splits.
    5. Render reliability diagrams.
    6. Write calibration record + report.
    """
    root = (project_root or find_project_root()).resolve()
    config = load_training_config(config_path, project_root=root)
    set_seed(config.seed)

    device = resolve_device()
    use_amp = should_use_amp(device, config.mixed_precision)

    logger.info("Loading checkpoint: %s", checkpoint_path)
    model, checkpoint_payload = load_checkpoint_model(checkpoint_path, device)

    # Read temperature stored in checkpoint (set by temperature_scaling.py)
    checkpoint_T = checkpoint_payload.get("temperature", None)
    if checkpoint_T is not None:
        logger.info("Checkpoint carries pre-fitted temperature T=%.6f", checkpoint_T)
    else:
        logger.info("No temperature in checkpoint; will fit on val split.")

    # ── collect logits ──────────────────────────────────────────────────────
    split_logits: dict[str, np.ndarray] = {}
    split_labels: dict[str, np.ndarray] = {}

    split_names = {"val": "val.csv", "test_seen": "test_seen.csv", "test_unseen": "test_unseen.csv"}
    for split_name, csv_name in split_names.items():
        csv_path = config.split_dir / csv_name
        loader = build_loader(csv_path, config, device)
        if loader is None:
            split_logits[split_name] = np.array([])
            split_labels[split_name] = np.array([])
            logger.warning("Split %s: no data available.", split_name)
        else:
            lgts, lbls = collect_logits(model, loader, device, use_amp=use_amp)
            split_logits[split_name] = lgts
            split_labels[split_name] = lbls
            logger.info("Split %s: %d samples collected.", split_name, len(lgts))

    # ── fit / use temperature on VALIDATION ONLY ────────────────────────────
    val_logits = split_logits["val"]
    val_labels = split_labels["val"]

    if checkpoint_T is not None:
        T = float(checkpoint_T)
        fit_info: dict[str, Any] = checkpoint_payload.get(
            "calibration_fit_info",
            {"T": T, "source": "loaded_from_checkpoint"},
        )
    elif val_logits.size == 0:
        logger.error("Validation split is empty — cannot fit temperature. Using T=1.0.")
        T = 1.0
        fit_info = {"T": 1.0, "warning": "empty_val_split"}
    else:
        T, fit_info = fit_temperature(val_logits, val_labels)

    # ── metrics before calibration (T=1) ───────────────────────────────────
    before_metrics: dict[str, Any] = {}
    for sp in split_names:
        logits = split_logits[sp]
        labels = split_labels[sp]
        if logits.size == 0:
            before_metrics[sp] = {"support": 0, "note": "empty_split"}
        else:
            probs = 1.0 / (1.0 + np.exp(-logits))
            before_metrics[sp] = full_calibration_metrics(labels, probs, n_bins=n_bins)

    # ── metrics after calibration ───────────────────────────────────────────
    after_metrics: dict[str, Any] = {}
    for sp in split_names:
        logits = split_logits[sp]
        labels = split_labels[sp]
        if logits.size == 0:
            after_metrics[sp] = {"support": 0, "note": "empty_split"}
        else:
            cal_probs = 1.0 / (1.0 + np.exp(-logits / max(T, 1e-4)))
            after_metrics[sp] = full_calibration_metrics(labels, cal_probs, n_bins=n_bins)

    # ── reliability diagrams ────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    if val_logits.size > 0:
        bm = before_metrics["val"]
        rd = bm.get("reliability_diagram", {})
        _draw_reliability_diagram(
            np.array(rd.get("bin_confs", [])),
            np.array(rd.get("bin_accs", [])),
            np.array(rd.get("bin_counts", []), dtype=int),
            bm.get("ece", 0.0),
            title="Reliability Diagram — Before Calibration (val split)",
            output_path=output_dir / "reliability_before.png",
        )

        am = after_metrics["val"]
        rd2 = am.get("reliability_diagram", {})
        _draw_reliability_diagram(
            np.array(rd2.get("bin_confs", [])),
            np.array(rd2.get("bin_accs", [])),
            np.array(rd2.get("bin_counts", []), dtype=int),
            am.get("ece", 0.0),
            title=f"Reliability Diagram — After Calibration (T={T:.4f}, val split)",
            output_path=output_dir / "reliability_after.png",
        )
    else:
        logger.warning("Skipping reliability diagrams — validation split is empty.")

    # ── save calibration record ─────────────────────────────────────────────
    git_commit = get_git_commit(root)

    cal_record = CalibrationRecord(
        temperature=T,
        fit_info=fit_info,
        fitted_on_split="val",
        source_checkpoint=str(checkpoint_path.resolve()),
    )
    cal_record_path = output_dir / "calibration_record.json"
    save_calibration(cal_record, cal_record_path)

    # ── assemble result dict ────────────────────────────────────────────────
    result: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "config_path": str(config_path.resolve()),
        "git_commit": git_commit,
        "random_seed": config.seed,
        "temperature": T,
        "calibration_record": {
            "temperature": T,
            "fitted_on_split": "val",
            "fit_info": fit_info,
            "source_checkpoint": str(checkpoint_path.resolve()),
        },
        "before_calibration": before_metrics,
        "after_calibration": after_metrics,
        "n_bins": n_bins,
        "note": (
            "Temperature fitted on validation split ONLY. "
            "Test sets were NEVER used to fit calibration parameters."
        ),
    }

    # ── write report ────────────────────────────────────────────────────────
    md_text = build_markdown_report(result, output_dir)
    md_path = output_dir / "calibration_report.md"
    md_path.write_text(md_text, encoding="utf-8")
    logger.info("Calibration report → %s", md_path)

    # ── write JSON ──────────────────────────────────────────────────────────
    json_path = output_dir / "calibration_result.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
        fh.write("\n")
    logger.info("Calibration JSON → %s", json_path)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AEGIS Audio Calibration Evaluator — ECE/Brier/NLL/AUC/F1 + reliability diagrams."
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to a trained audio model checkpoint (.pt).",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to audio_baseline.yaml (default: configs/audio_baseline.yaml).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for calibration outputs (default: reports/audio/calibration).",
    )
    parser.add_argument(
        "--project-root", type=Path, default=None,
        help="AEGIS project root (auto-discovered if omitted).",
    )
    parser.add_argument(
        "--n-bins", type=int, default=10,
        help="Number of bins for ECE / reliability diagram (default: 10).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)

    root = (args.project_root or find_project_root()).resolve()
    config_path = (args.config or root / "configs" / "audio_baseline.yaml").resolve()
    output_dir = (args.output_dir or root / "reports" / "audio" / "calibration").resolve()

    result = run_calibration_evaluation(
        checkpoint_path=args.checkpoint.resolve(),
        config_path=config_path,
        output_dir=output_dir,
        project_root=root,
        n_bins=args.n_bins,
    )

    T = result["temperature"]
    print("\n" + "=" * 60)
    print("AEGIS AUDIO CALIBRATION — SUMMARY")
    print("=" * 60)
    print(f"  Temperature T        : {T:.6f}")
    print(f"  (T>1=overconfident, T<1=underconfident, T=1=calibrated)")
    print("")
    for split_name in ("val", "test_seen", "test_unseen"):
        b = result["before_calibration"].get(split_name, {})
        a = result["after_calibration"].get(split_name, {})
        n = b.get("support", 0)
        if n == 0:
            print(f"  [{split_name:12s}] empty — skipped")
            continue
        b_ece = b.get("ece")
        a_ece = a.get("ece")
        b_nll = b.get("nll")
        a_nll = a.get("nll")
        ece_str = f"ECE {b_ece:.4f} → {a_ece:.4f}" if b_ece is not None else "ECE N/A"
        nll_str = f"NLL {b_nll:.4f} → {a_nll:.4f}" if b_nll is not None else "NLL N/A"
        print(f"  [{split_name:12s}] {ece_str}  |  {nll_str}  (n={n})")
    print("")
    print(f"  Outputs → {output_dir}")
    print("=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
