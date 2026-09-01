"""AEGIS Generalization Gap Analysis Module for Audio.

Reads experiment result JSON files produced by src/audio/training/evaluate.py
and produces a structured comparison between models, answering the core
research question:

    Does spectral/phonetic evidence improve robustness to *unseen* TTS/voice cloning generators,
    rather than merely improving seen-generator accuracy?

Usage (CLI)::

    python -m src.audio.analysis.generalization \\
        --baseline    results/audio_baseline.json \\
        --output-dir  reports/audio

All numbers are read from disk — none are hard-coded.

Outputs
-------
reports/audio/generalization_report.json   machine-readable comparison artifact
reports/audio/generalization_report.md     human-readable research report with ASCII charts
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────────

METRICS = ("accuracy", "f1", "roc_auc", "balanced_accuracy", "precision", "recall")


@dataclass
class SplitMetrics:
    """Metrics for one evaluation split."""
    accuracy: float | None = None
    f1: float | None = None
    roc_auc: float | None = None
    balanced_accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    support: int = 0
    confusion_matrix: list[list[int]] = field(default_factory=lambda: [[0, 0], [0, 0]])
    note: str = ""

    def get(self, metric: str) -> float | None:
        return getattr(self, metric, None)


@dataclass
class ModelProfile:
    """Extracted profile for a single model across all splits."""
    label: str
    result_path: str
    evaluated_at: str
    git_commit: str | None
    split_version: str
    preprocessing_version: str
    random_seed: int | None
    model_architecture: dict[str, Any]
    val: SplitMetrics
    seen: SplitMetrics
    unseen: SplitMetrics
    gap: dict[str, float | None]          # seen_metric – unseen_metric per metric
    gap_computable: bool
    gap_note: str


@dataclass
class ComparisonResult:
    """Cross-model comparison artifact."""
    generated_at: str
    models: list[ModelProfile]
    best_seen: dict[str, str]             # metric → model label
    best_unseen: dict[str, str]
    smallest_gap: dict[str, str]
    spectral_improves_seen: dict[str, bool | None]
    spectral_improves_unseen: dict[str, bool | None]
    spectral_narrows_gap: dict[str, bool | None]
    analysis_notes: list[str]


# ──────────────────────────────────────────────────────────────────────────────
# Parsing
# ──────────────────────────────────────────────────────────────────────────────

def _parse_split(raw: dict[str, Any]) -> SplitMetrics:
    return SplitMetrics(
        accuracy=raw.get("accuracy"),
        f1=raw.get("f1"),
        roc_auc=raw.get("roc_auc"),
        balanced_accuracy=raw.get("balanced_accuracy"),
        precision=raw.get("precision"),
        recall=raw.get("recall"),
        support=int(raw.get("support", 0)),
        confusion_matrix=raw.get("confusion_matrix", [[0, 0], [0, 0]]),
        note=raw.get("note", raw.get("roc_auc_note", "")),
    )


def _derive_gap(seen: SplitMetrics, unseen: SplitMetrics) -> tuple[dict[str, float | None], bool, str]:
    """Compute seen – unseen per metric. Returns (gap_dict, computable, note)."""
    if unseen.support == 0:
        return {m: None for m in METRICS}, False, "test_unseen split is empty; gap is not computable yet."
    gap: dict[str, float | None] = {}
    for m in METRICS:
        sv = seen.get(m)
        uv = unseen.get(m)
        if sv is None or uv is None:
            gap[m] = None
        else:
            gap[m] = round(float(sv) - float(uv), 6)
    return gap, True, ""


def load_model_profile(result_path: Path, label: str) -> ModelProfile:
    """Load and validate an experiment result JSON into a ModelProfile."""
    if not result_path.is_file():
        raise FileNotFoundError(f"Result file not found: {result_path}")

    with result_path.open(encoding="utf-8") as fh:
        raw: dict[str, Any] = json.load(fh)

    splits_raw = raw.get("splits", {})
    val = _parse_split(splits_raw.get("val", {}))
    seen = _parse_split(splits_raw.get("test_seen", {}))
    unseen = _parse_split(splits_raw.get("test_unseen", {}))

    # Re-derive gap from the raw split values so this module is the authority
    gap, computable, gap_note = _derive_gap(seen, unseen)

    return ModelProfile(
        label=label,
        result_path=str(result_path.resolve()),
        evaluated_at=raw.get("evaluated_at", ""),
        git_commit=raw.get("git_commit"),
        split_version=raw.get("split_version", ""),
        preprocessing_version=raw.get("preprocessing_version", ""),
        random_seed=raw.get("random_seed"),
        model_architecture=raw.get("model_architecture", {}),
        val=val,
        seen=seen,
        unseen=unseen,
        gap=gap,
        gap_computable=computable,
        gap_note=gap_note,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Comparison logic
# ──────────────────────────────────────────────────────────────────────────────

def _best_label(profiles: list[ModelProfile], split: str, metric: str) -> str:
    """Return label of model with highest value on (split, metric). 'N/A' if none available."""
    best_val: float | None = None
    best_label = "N/A"
    for p in profiles:
        split_obj = {"val": p.val, "seen": p.seen, "unseen": p.unseen}.get(split)
        if split_obj is None:
            continue
        v = split_obj.get(metric)
        if v is not None and (best_val is None or v > best_val):
            best_val = v
            best_label = p.label
    return best_label


def _smallest_gap_label(profiles: list[ModelProfile], metric: str) -> str:
    """Return label of model with smallest *non-negative* generalization gap."""
    best_abs: float | None = None
    best_label = "N/A"
    for p in profiles:
        g = p.gap.get(metric)
        if g is not None and (best_abs is None or abs(g) < best_abs):
            best_abs = abs(g)
            best_label = p.label
    return best_label


def compare_models(profiles: list[ModelProfile]) -> ComparisonResult:
    """Run the full cross-model comparison."""
    if len(profiles) < 1:
        raise ValueError("Need at least 1 model profile to analyze.")

    # For audio, we currently have just baseline - can be extended for ablation studies
    baseline = profiles[0]
    spectral = profiles[1] if len(profiles) > 1 else None

    best_seen: dict[str, str] = {}
    best_unseen: dict[str, str] = {}
    smallest_gap: dict[str, str] = {}
    spectral_improves_seen: dict[str, bool | None] = {}
    spectral_improves_unseen: dict[str, bool | None] = {}
    spectral_narrows_gap: dict[str, bool | None] = {}

    for metric in METRICS:
        best_seen[metric] = _best_label(profiles, "seen", metric)
        best_unseen[metric] = _best_label(profiles, "unseen", metric)
        smallest_gap[metric] = _smallest_gap_label(profiles, metric)

        # Spectral comparison (baseline vs spectral) if available
        if spectral:
            bs = baseline.seen.get(metric)
            ss = spectral.seen.get(metric)
            bu = baseline.unseen.get(metric)
            su = spectral.unseen.get(metric)
            bg = baseline.gap.get(metric)
            sg = spectral.gap.get(metric)

            spectral_improves_seen[metric] = (ss > bs) if (ss is not None and bs is not None) else None
            spectral_improves_unseen[metric] = (su > bu) if (su is not None and bu is not None) else None
            # Smaller gap is better (closer to 0)
            spectral_narrows_gap[metric] = (abs(sg) < abs(bg)) if (sg is not None and bg is not None) else None
        else:
            spectral_improves_seen[metric] = None
            spectral_improves_unseen[metric] = None
            spectral_narrows_gap[metric] = None

    # Build analysis notes
    notes: list[str] = []
    for p in profiles:
        if not p.gap_computable:
            notes.append(f"[{p.label}] {p.gap_note}")

    # Core scientific question answer
    if spectral:
        unseen_improvable = any(v is True for v in spectral_improves_unseen.values())
        gap_narrowable = any(v is True for v in spectral_narrows_gap.values())
        if unseen_improvable and gap_narrowable:
            notes.append(
                "FINDING: Spectral/phonetic evidence shows improvements on both unseen performance "
                "and generalization gap for at least one metric."
            )
        elif all(v is None for v in spectral_improves_unseen.values()):
            notes.append(
                "FINDING: Cannot determine spectral robustness effect — unseen split is empty. "
                "Run preprocessing on unseen generator samples and re-evaluate."
            )
        else:
            notes.append(
                "FINDING: Based on available data, spectral features do not demonstrably reduce the "
                "generalization gap. More unseen data is required for a confident conclusion."
            )
    else:
        notes.append(
            "FINDING: Single model analysis — baseline performance only. "
            "Add spectral/phonetic ablation models for comparison."
        )

    return ComparisonResult(
        generated_at=datetime.now(timezone.utc).isoformat(),
        models=profiles,
        best_seen=best_seen,
        best_unseen=best_unseen,
        smallest_gap=smallest_gap,
        spectral_improves_seen=spectral_improves_seen,
        spectral_improves_unseen=spectral_improves_unseen,
        spectral_narrows_gap=spectral_narrows_gap,
        analysis_notes=notes,
    )


# ──────────────────────────────────────────────────────────────────────────────
# ASCII chart helpers
# ──────────────────────────────────────────────────────────────────────────────

def _bar(value: float | None, width: int = 20, fill: str = "█") -> str:
    """Render a proportional ASCII bar; value in [0, 1]."""
    if value is None:
        return "─" * (width // 4) + " (N/A)"
    filled = round(value * width)
    return fill * filled + " " * (width - filled) + f" {value:.4f}"


def _performance_chart(profiles: list[ModelProfile], metric: str, width: int = 20) -> str:
    """Two-row bar chart per model: seen vs unseen."""
    lines = [
        f"Performance (metric={metric})",
        "│",
    ]
    for p in profiles:
        sv = p.seen.get(metric)
        uv = p.unseen.get(metric)
        lines.append(f"│  [{p.label}]")
        lines.append(f"│    Seen  : {_bar(sv, width, '█')}")
        lines.append(f"│    Unseen: {_bar(uv, width, '░')}")
        lines.append("│")
    lines.append("└" + "─" * (width + 30))
    return "\n".join(lines)


def _gap_chart(profiles: list[ModelProfile], metric: str, width: int = 20) -> str:
    """Bar chart of generalization gaps per model."""
    lines = [
        f"Generalization Gap (metric={metric}; defined as seen – unseen)",
        "│",
    ]
    max_gap = 1.0  # normalise bars against unit scale
    for p in profiles:
        g = p.gap.get(metric)
        if g is not None:
            max_gap = max(max_gap, abs(g))
    for p in profiles:
        g = p.gap.get(metric)
        if g is not None:
            norm = min(abs(g) / max_gap, 1.0)
            bar_str = "█" * round(norm * width)
            lines.append(f"│  [{p.label}]  {bar_str} {g:+.4f}")
        else:
            lines.append(f"│  [{p.label}]  {'─' * (width // 4)} (N/A — unseen split empty)")
        lines.append("│")
    lines.append("└" + "─" * (width + 30))
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Report writers
# ──────────────────────────────────────────────────────────────────────────────

def _fmt(v: float | None, decimals: int = 4) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}"


def _bool_str(v: bool | None) -> str:
    if v is None:
        return "Indeterminate (unseen split empty)"
    return "Yes ✓" if v else "No ✗"


def _architecture_summary(arch: dict[str, Any]) -> str:
    """One-line architecture string."""
    mtype = arch.get("model_type", "baseline")
    if mtype == "mel_cnn":
        n_mels = arch.get("n_mels", 128)
        hidden = arch.get("hidden_dim", 256)
        blocks = arch.get("num_conv_blocks", 4)
        return f"MelCNN(n_mels={n_mels}, hidden={hidden}, blocks={blocks})"
    elif mtype == "wav2vec2_cnn":
        return f"Wav2Vec2CNN(backbone={arch.get('backbone', 'base')})"
    return f"Baseline({mtype})"


def _confusion_table(cm: list[list[int]]) -> str:
    """Mini confusion matrix as inline text."""
    if len(cm) < 2:
        return "unavailable"
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]
    return (
        f"  TN={tn}  FP={fp}\n"
        f"  FN={fn}  TP={tp}"
    )


def build_markdown_report(result: ComparisonResult) -> str:
    """Generate the full Markdown research report."""
    baseline = result.models[0]
    spectral = result.models[1] if len(result.models) > 1 else None

    lines: list[str] = [
        "# AEGIS Audio Generalization Gap Analysis Report",
        "",
        f"> Generated at: `{result.generated_at}`",
        "",
        "---",
        "",
        "## 1. Overview",
        "",
        "This report answers the central AEGIS audio research question:",
        "",
        "> **Does spectral/phonetic evidence improve robustness to unseen TTS/voice cloning generators,**",
        "> **rather than merely improving seen-generator accuracy?**",
        "",
        "Model(s) analyzed:",
        "",
    ]

    for p in result.models:
        lines.append(f"- **{p.label}**: {_architecture_summary(p.model_architecture)}")
        lines.append(f"  - Result file: `{p.result_path}`")
        lines.append(f"  - Evaluated at: `{p.evaluated_at}`")
        lines.append(f"  - Git commit: `{p.git_commit or 'unavailable'}`")
        lines.append(f"  - Split version: `{p.split_version}`, Preprocessing: `{p.preprocessing_version}`")
        lines.append(f"  - Random seed: `{p.random_seed}`")
        lines.append("")

    lines += [
        "---",
        "",
        "## 2. Per-Split Metrics",
        "",
    ]

    for p in result.models:
        lines.append(f"### {p.label}")
        lines.append("")
        lines.append("| Split | Accuracy | F1 | ROC-AUC | Balanced Acc | Support |")
        lines.append("|-------|----------|----|---------|--------------|---------| ")
        for split_name, sm in [("val", p.val), ("test_seen", p.seen), ("test_unseen", p.unseen)]:
            lines.append(
                f"| {split_name} "
                f"| {_fmt(sm.accuracy)} "
                f"| {_fmt(sm.f1)} "
                f"| {_fmt(sm.roc_auc)} "
                f"| {_fmt(sm.balanced_accuracy)} "
                f"| {sm.support} |"
            )
            if sm.note:
                lines.append(f"|  | *{sm.note}* | | | | |")
        lines.append("")

        lines.append("**Confusion matrices (test_seen):**")
        lines.append("```")
        lines.append("         Predicted")
        lines.append("          Real  Fake")
        lines.append(_confusion_table(p.seen.confusion_matrix))
        lines.append("```")
        lines.append("")

        if p.gap_computable:
            lines.append("**Generalization gap (seen − unseen):**")
            lines.append("")
            lines.append("| Metric | Gap |")
            lines.append("|--------|-----|")
            for m in METRICS:
                lines.append(f"| {m} | {_fmt(p.gap.get(m))} |")
        else:
            lines.append(f"> ⚠️ **Gap not computable**: {p.gap_note}")
        lines.append("")

    lines += [
        "---",
        "",
        "## 3. ASCII Visualisations",
        "",
        "### 3.1 Performance Chart",
        "",
    ]

    for metric in ("accuracy", "f1", "roc_auc", "balanced_accuracy"):
        lines.append(f"#### {metric}")
        lines.append("```")
        lines.append(_performance_chart(result.models, metric))
        lines.append("```")
        lines.append("")

    lines += [
        "### 3.2 Generalization Gap Chart",
        "",
    ]

    for metric in ("accuracy", "f1", "roc_auc", "balanced_accuracy"):
        lines.append(f"#### {metric}")
        lines.append("```")
        lines.append(_gap_chart(result.models, metric))
        lines.append("```")
        lines.append("")

    lines += [
        "---",
        "",
        "## 4. Research Questions",
        "",
    ]

    # Q1
    lines.append("### Q1 — Which model performs best on seen data?")
    lines.append("")
    lines.append("| Metric | Best Model (Seen) |")
    lines.append("|--------|-------------------|")
    for m in METRICS:
        lines.append(f"| {m} | {result.best_seen.get(m, 'N/A')} |")
    lines.append("")

    # Q2
    lines.append("### Q2 — Which model performs best on unseen data?")
    lines.append("")
    lines.append("| Metric | Best Model (Unseen) |")
    lines.append("|--------|---------------------|")
    for m in METRICS:
        lines.append(f"| {m} | {result.best_unseen.get(m, 'N/A')} |")
    lines.append("")

    # Q3
    lines.append("### Q3 — Which model has the smallest generalization gap?")
    lines.append("")
    lines.append("| Metric | Smallest Gap Model |")
    lines.append("|--------|--------------------|")
    for m in METRICS:
        lines.append(f"| {m} | {result.smallest_gap.get(m, 'N/A')} |")
    lines.append("")

    # Q4
    lines.append("### Q4 — Does spectral/phonetic analysis improve robustness (unseen performance)?")
    lines.append("")
    if spectral:
        lines.append("| Metric | Spectral improves Unseen? |")
        lines.append("|--------|---------------------------|")
        for m in METRICS:
            lines.append(f"| {m} | {_bool_str(result.spectral_improves_unseen.get(m))} |")
    else:
        lines.append("*Single model analysis — spectral comparison not available.*")
    lines.append("")

    # Q5
    lines.append("### Q5 — Is any improvement merely due to better seen performance?")
    lines.append("")
    if spectral:
        lines.append(
            "If spectral features improve seen performance **without** narrowing the gap, the benefit "
            "is superficial (better fit to seen generators, not better generalization)."
        )
        lines.append("")
        lines.append("| Metric | Spectral improves Seen? | Spectral narrows Gap? | Assessment |")
        lines.append("|--------|------------------------|----------------------|------------|")
        for m in METRICS:
            imp_seen = result.spectral_improves_seen.get(m)
            narrows = result.spectral_narrows_gap.get(m)
            if imp_seen is None:
                assessment = "Indeterminate"
            elif imp_seen and narrows:
                assessment = "Genuine generalization improvement"
            elif imp_seen and narrows is False:
                assessment = "Seen-only improvement (superficial)"
            elif imp_seen is False and narrows:
                assessment = "Gap narrows despite weaker seen perf."
            elif imp_seen is False and narrows is False:
                assessment = "No benefit in either dimension"
            else:
                assessment = "Indeterminate"
            lines.append(
                f"| {m} | {_bool_str(imp_seen)} | {_bool_str(narrows)} | {assessment} |"
            )
    else:
        lines.append("*Single model analysis — comparison not available.*")
    lines.append("")

    # Q6
    lines.append("### Q6 — Where does the model fail?")
    lines.append("")
    for p in result.models:
        lines.append(f"**{p.label}** (test_seen confusion matrix):")
        lines.append("```")
        lines.append("         Predicted")
        lines.append("          Real  Fake")
        lines.append(_confusion_table(p.seen.confusion_matrix))
        lines.append("```")
        cm = p.seen.confusion_matrix
        if len(cm) >= 2:
            tn = cm[0][0]; fp = cm[0][1]
            fn = cm[1][0]; tp = cm[1][1]
            total = tn + fp + fn + tp
            if total > 0:
                if fp > fn:
                    lines.append(
                        f"→ **FP-heavy**: model over-predicts fake "
                        f"(FP={fp}, FN={fn}). Risk: flagging real audio incorrectly."
                    )
                elif fn > fp:
                    lines.append(
                        f"→ **FN-heavy**: model misses fakes "
                        f"(FP={fp}, FN={fn}). Risk: fake audio passing undetected."
                    )
                else:
                    lines.append(f"→ FP={fp}, FN={fn} — balanced errors.")
        if p.unseen.support == 0:
            lines.append(
                "→ **Unseen split is empty** — failure modes on novel TTS/voice cloning generators "
                "cannot be characterised until unseen test data is preprocessed."
            )
        lines.append("")

    lines += [
        "---",
        "",
        "## 5. Analysis Notes",
        "",
    ]
    for note in result.analysis_notes:
        lines.append(f"- {note}")
    lines.append("")

    lines += [
        "---",
        "",
        "## 6. Limitations & Next Steps",
        "",
        (
            "- The `test_unseen` split is currently **empty** because no unseen-generator "
            "audio has been preprocessed. All gap metrics are therefore `null` and "
            "cannot support any conclusion about spectral robustness yet.\n"
            "- Recommended next step: preprocess a held-out dataset from a TTS/voice cloning "
            "family not represented in the training or seen-test sets (e.g., ElevenLabs, XTTS-v2, StyleTTS 2), "
            "re-run evaluation, and re-execute this analysis script.\n"
            "- This module is designed to be re-run without modification once the unseen "
            "split is populated. Results will update automatically."
        ),
        "",
        "---",
        "*AEGIS Audio Generalization Gap Analysis — automated, reproducible, no hard-coded numbers.*",
    ]

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# JSON serialisation helper
# ──────────────────────────────────────────────────────────────────────────────

def _to_json_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_serializable(v) for v in obj]
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def build_json_report(result: ComparisonResult) -> dict[str, Any]:
    """Produce a machine-readable dict from the comparison result."""
    models_json = []
    for p in result.models:
        models_json.append({
            "label": p.label,
            "result_path": p.result_path,
            "evaluated_at": p.evaluated_at,
            "git_commit": p.git_commit,
            "split_version": p.split_version,
            "preprocessing_version": p.preprocessing_version,
            "random_seed": p.random_seed,
            "model_architecture": p.model_architecture,
            "splits": {
                "val": asdict(p.val),
                "test_seen": asdict(p.seen),
                "test_unseen": asdict(p.unseen),
            },
            "generalization_gap": {
                "computable": p.gap_computable,
                "note": p.gap_note,
                "metrics": p.gap,
            },
        })

    return {
        "generated_at": result.generated_at,
        "models": models_json,
        "best_seen": result.best_seen,
        "best_unseen": result.best_unseen,
        "smallest_gap": result.smallest_gap,
        "spectral_improves_seen": {k: v for k, v in result.spectral_improves_seen.items()},
        "spectral_improves_unseen": {k: v for k, v in result.spectral_improves_unseen.items()},
        "spectral_narrows_gap": {k: v for k, v in result.spectral_narrows_gap.items()},
        "analysis_notes": result.analysis_notes,
    }


# ──────────────────────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────────────────────

def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(_to_json_serializable(payload), fh, indent=2)
        fh.write("\n")
    tmp.replace(path)
    logger.info("Wrote JSON report: %s", path)


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    logger.info("Wrote Markdown report: %s", path)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _find_project_root() -> Path:
    """Walk up from this file looking for configs/ directory."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "configs").is_dir():
            return parent
    return Path.cwd()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AEGIS Audio Generalization Gap Analysis — compares experiment result JSONs."
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Path to baseline model result JSON (default: results/audio_baseline.json).",
    )
    parser.add_argument(
        "--spectral",
        type=Path,
        default=None,
        help="Path to spectral/phonetic enhancement result JSON (optional).",
    )
    parser.add_argument(
        "--baseline-label",
        default="Audio Baseline",
        help="Display label for the baseline model.",
    )
    parser.add_argument(
        "--spectral-label",
        default="Spectral Enhanced",
        help="Display label for the spectral model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output reports (default: reports/audio/).",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="AEGIS project root (auto-discovered if omitted).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)

    root = (args.project_root or _find_project_root()).resolve()
    baseline_path = (args.baseline or root / "results" / "audio_baseline.json").resolve()
    spectral_path = args.spectral
    output_dir = (args.output_dir or root / "reports" / "audio").resolve()

    logger.info("Project root  : %s", root)
    logger.info("Baseline file : %s", baseline_path)
    logger.info("Spectral file : %s", spectral_path)
    logger.info("Output dir    : %s", output_dir)

    baseline_profile = load_model_profile(baseline_path, args.baseline_label)
    profiles = [baseline_profile]
    
    if spectral_path:
        spectral_profile = load_model_profile(spectral_path, args.spectral_label)
        profiles.append(spectral_profile)

    result = compare_models(profiles)

    md_text = build_markdown_report(result)
    json_payload = build_json_report(result)

    md_path = output_dir / "generalization_report.md"
    json_path = output_dir / "generalization_report.json"

    write_markdown(md_path, md_text)
    write_json(json_path, json_payload)

    # Print key findings to stdout
    print("\n" + "=" * 60)
    print("AEGIS AUDIO GENERALIZATION GAP ANALYSIS — KEY FINDINGS")
    print("=" * 60)
    for note in result.analysis_notes:
        print(f"  • {note}")
    print("")
    print(f"  Reports written to:")
    print(f"    {md_path}")
    print(f"    {json_path}")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
