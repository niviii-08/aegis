"""Temperature Scaling for AEGIS audio binary classifiers.

Theory
------
Temperature scaling (Guo et al., 2017) is the simplest post-hoc calibration
method.  The raw logit ``z`` from the model is divided by a single learned
scalar T > 0 before applying sigmoid:

    P_calibrated = sigmoid(z / T)

When T > 1  the distribution is softened (model was overconfident).
When T < 1  the distribution is sharpened (model was underconfident).
When T = 1  calibration leaves predictions unchanged.

The scalar T is optimised by minimising the *negative log-likelihood* (NLL)
on a **calibration / validation set only**.  It must NEVER be fitted on the
final test set — that would constitute test-set leakage.

Distinction: probability vs. confidence vs. conformal prediction
----------------------------------------------------------------
* ``probability``   — sigmoid(z / T).  A calibrated estimate of P(fake).
* ``confidence``    — *not* a probability.  Reserved for a future model-level
                      reliability score (e.g. ensemble agreement).
* ``conformal prediction`` — when implemented, will produce prediction sets with
                      a guaranteed coverage guarantee (e.g. 90%).  This is
                      distinct from calibrated probabilities and must be
                      represented separately in the prediction API output.

Prediction API structure (current)::

    {
        "label":       "fake",
        "probability": 0.73,
        "calibrated":  true,
        "temperature": 1.42,
        "note":        "probability is P(fake); see docs for confidence/conformal distinction"
    }

References
----------
Guo, C. et al. (2017). On calibration of modern neural networks. ICML.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core layer
# ─────────────────────────────────────────────────────────────────────────────

class TemperatureScalingLayer(nn.Module):
    """Single-parameter post-hoc calibration layer.

    Parameters
    ----------
    init_temperature:
        Starting value for T.  Optimisation will update this.
    """

    def __init__(self, init_temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = nn.Parameter(
            torch.tensor([init_temperature], dtype=torch.float32)
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Divide logits by temperature; return scaled logits (not probabilities).

        The caller applies ``sigmoid`` to obtain calibrated P(fake).
        """
        return logits / self.temperature.clamp(min=1e-4)

    def calibrated_proba(self, logits: torch.Tensor) -> torch.Tensor:
        """Convenience: return P(fake) after temperature scaling."""
        return torch.sigmoid(self.forward(logits))

    @property
    def T(self) -> float:
        return float(self.temperature.item())


# ─────────────────────────────────────────────────────────────────────────────
# Fitting — uses scipy because it's simpler and more reliable for a 1-D search
# ─────────────────────────────────────────────────────────────────────────────

def _nll(temperature: float, logits: np.ndarray, labels: np.ndarray) -> float:
    """Binary NLL for a candidate temperature value."""
    eps = 1e-7
    scaled = logits / max(temperature, eps)
    probs = 1.0 / (1.0 + np.exp(-scaled))
    probs = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs)))


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    bounds: tuple[float, float] = (0.01, 10.0),
) -> tuple[float, dict[str, Any]]:
    """Fit temperature T using NLL minimisation on (logits, labels).

    IMPORTANT: logits and labels MUST come from the validation/calibration set.
    Never pass test-set data here.

    Returns
    -------
    temperature : float
        Optimal T that minimises NLL on the calibration set.
    fit_info : dict
        Diagnostic metadata (pre/post NLL, convergence flag, etc.).
    """
    logits = np.asarray(logits, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()

    if logits.size == 0:
        raise ValueError("Cannot fit temperature: logits array is empty.")
    if len(np.unique(labels)) < 2:
        logger.warning(
            "Calibration set has only one class — temperature scaling "
            "cannot be meaningfully fitted. Returning T=1.0."
        )
        return 1.0, {"warning": "single_class_calibration_set", "T": 1.0}

    nll_before = _nll(1.0, logits, labels)

    result = minimize_scalar(
        _nll,
        args=(logits, labels),
        bounds=bounds,
        method="bounded",
        options={"xatol": 1e-5, "maxiter": 500},
    )

    T_opt = float(result.x)
    nll_after = float(result.fun)

    fit_info: dict[str, Any] = {
        "T": T_opt,
        "nll_before_calibration": round(nll_before, 6),
        "nll_after_calibration": round(nll_after, 6),
        "nll_reduction": round(nll_before - nll_after, 6),
        "converged": bool(result.success),
        "optimizer_message": str(result.message) if hasattr(result, "message") else "",
        "n_calibration_samples": int(logits.size),
        "calibration_set": "validation (MUST NOT be test set)",
    }

    logger.info(
        "Temperature fitted: T=%.4f  NLL: %.4f → %.4f  (Δ=%.4f)  n=%d",
        T_opt, nll_before, nll_after, nll_before - nll_after, logits.size,
    )
    return T_opt, fit_info


# ─────────────────────────────────────────────────────────────────────────────
# Calibrated model wrapper — drop-in for streaming inference
# ─────────────────────────────────────────────────────────────────────────────

class CalibratedModel(nn.Module):
    """Wraps any audio backbone and applies temperature scaling on its logit.

    This wrapper is fully drop-in compatible with ``AudioStreamingInference``
    and with the evaluate_checkpoint pipeline: it exposes the same ``forward``
    signature as the base model and a ``describe()`` method.

    Usage::

        base_model = ...           # loaded checkpoint
        cal_model = CalibratedModel(base_model, temperature=1.42)
        # forward() now returns temperature-scaled logits
        logits = cal_model(batch_x)
    """

    def __init__(self, base_model: nn.Module, temperature: float) -> None:
        super().__init__()
        self.base_model = base_model
        self.temperature_layer = TemperatureScalingLayer(temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.base_model(x)
        return self.temperature_layer(logits)

    def describe(self) -> dict[str, Any]:
        base_desc = self.base_model.describe() if hasattr(self.base_model, "describe") else {}
        return {
            **base_desc,
            "calibrated": True,
            "temperature": self.temperature_layer.T,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint save / load
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CalibrationRecord:
    """Everything needed to reproduce calibrated predictions."""
    temperature: float
    fit_info: dict[str, Any]
    fitted_on_split: str            # e.g. "val"
    source_checkpoint: str          # path to the base model checkpoint
    aegis_version: str = "1.0.0"
    note: str = (
        "T is learned on validation data only. "
        "probability = sigmoid(logit / T). "
        "This is NOT a confidence interval. "
        "Conformal prediction sets (when implemented) will be separate."
    )


def save_calibration(record: CalibrationRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(asdict(record), fh, indent=2)
        fh.write("\n")
    tmp.replace(path)
    logger.info("Calibration record saved → %s", path)


def load_calibration(path: Path) -> CalibrationRecord:
    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    return CalibrationRecord(**raw)


def save_calibrated_checkpoint(
    *,
    base_checkpoint_path: Path,
    temperature: float,
    fit_info: dict[str, Any],
    output_path: Path,
) -> None:
    """Save a calibrated checkpoint that retains the full base model payload.

    The saved file is a standard torch checkpoint dict with an extra
    ``temperature`` key.  Loading via ``load_checkpoint_model`` in
    ``audio.training.evaluate`` will transparently apply temperature scaling
    when ``temperature != 1.0``.
    """
    payload = torch.load(base_checkpoint_path, map_location="cpu", weights_only=False)
    payload["temperature"] = temperature
    payload["calibration_fit_info"] = fit_info
    payload["calibrated"] = True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(output_path)
    logger.info("Calibrated checkpoint saved → %s", output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Prediction API helper  (identical to image counterpart)
# ─────────────────────────────────────────────────────────────────────────────

def make_prediction(
    logit: float,
    *,
    temperature: float = 1.0,
    threshold: float = 0.5,
    calibrated: bool = True,
) -> dict[str, Any]:
    """Return a structured prediction dict for a single sample.

    Terminology
    -----------
    probability
        Calibrated P(fake) in [0, 1].  This is *not* called a "confidence
        interval" — that term is reserved for statistical interval estimation.
    calibrated
        True when temperature scaling has been applied.
    confidence
        Intentionally absent from the current API.  A future module may add
        model-level reliability scores (e.g. ensemble disagreement, distance
        to training distribution).  These must NOT be conflated with
        calibrated probabilities.
    conformal_prediction_set
        Absent until conformal prediction is implemented.  When present it will
        carry a coverage guarantee and will be a separate top-level key.

    Example output::

        {
            "label":       "fake",
            "probability": 0.73,
            "calibrated":  true,
            "temperature": 1.42,
            "note": "..."
        }
    """
    prob = float(torch.sigmoid(torch.tensor(logit / max(temperature, 1e-4))).item())
    label = "fake" if prob >= threshold else "real"
    return {
        "label": label,
        "probability": round(prob, 6),
        "calibrated": calibrated,
        "temperature": round(temperature, 6) if calibrated else None,
        "note": (
            "probability is calibrated P(fake) via temperature scaling. "
            "This is not a confidence interval. "
            "Conformal prediction sets (coverage-guaranteed) are not yet implemented "
            "and will be represented as a separate key when added."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point: fit + save calibrated checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def _collect_logits_audio(
    model: nn.Module,
    config: Any,
    device: torch.device,
    *,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect (logits, labels) from the audio val split."""
    import sys
    from pathlib import Path

    _SRC_ROOT = Path(__file__).resolve().parents[2]
    if str(_SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(_SRC_ROOT))

    from torch.utils.data import DataLoader
    from audio.training.dataset import AudioDataset, resolve_samples

    val_csv = config.split_dir / "val.csv"
    if not val_csv.is_file():
        raise FileNotFoundError(f"Val split not found: {val_csv}")

    samples = resolve_samples(
        val_csv,
        config.metadata_path,
        config.project_root,
        preprocessing_version=config.preprocessing_version,
        feature_mode=config.feature_mode,
    )
    if not samples:
        raise RuntimeError("No preprocessed val samples found.")

    loader = DataLoader(
        AudioDataset(samples, feature_mode=config.feature_mode, is_training=False),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and device.type == "cuda",
    )

    model.eval()
    all_logits: list[float] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(batch_x)
            all_logits.extend(logits.cpu().float().numpy().tolist())
            all_labels.extend(batch_y.numpy().astype(int).tolist())

    return (
        np.array(all_logits, dtype=np.float64),
        np.array(all_labels, dtype=np.float64),
    )


def parse_args(argv: list[str] | None = None):
    import argparse
    p = argparse.ArgumentParser(
        description=(
            "AEGIS audio temperature scaling — fit T on val split and save "
            "models/audio/baseline_calibrated.pt."
        )
    )
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to base audio checkpoint (.pt).")
    p.add_argument("--config", type=Path, default=None,
                   help="Path to audio_baseline.yaml.")
    p.add_argument("--output", type=Path, default=None,
                   help="Output checkpoint path (default: models/audio/baseline_calibrated.pt).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Directory for calibration record JSON (default: reports/audio/calibration).")
    p.add_argument("--project-root", type=Path, default=None,
                   help="AEGIS project root (auto-discovered if omitted).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s %(message)s")

    import sys
    from pathlib import Path
    _SRC_ROOT = Path(__file__).resolve().parents[2]
    if str(_SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(_SRC_ROOT))

    from audio.training.evaluate import load_checkpoint_model
    from audio.training.utils import (
        find_project_root,
        load_training_config,
        resolve_device,
        set_seed,
        should_use_amp,
    )

    args = parse_args(argv)
    root = (args.project_root or find_project_root()).resolve()
    config_path = (args.config or root / "configs" / "audio_baseline.yaml").resolve()
    output_path = (args.output or root / "models" / "audio" / "baseline_calibrated.pt").resolve()
    output_dir = (args.output_dir or root / "reports" / "audio" / "calibration").resolve()

    config = load_training_config(config_path, root)
    set_seed(config.seed)
    device = resolve_device()
    use_amp = should_use_amp(device, config.mixed_precision)

    checkpoint_path = args.checkpoint.resolve()
    logger.info("Loading checkpoint: %s", checkpoint_path)
    model, _payload = load_checkpoint_model(checkpoint_path, device)

    logger.info("Collecting val logits...")
    logits, labels = _collect_logits_audio(model, config, device, use_amp=use_amp)
    logger.info("Val samples: %d", len(logits))

    T, fit_info = fit_temperature(logits, labels)

    # Save calibrated checkpoint
    save_calibrated_checkpoint(
        base_checkpoint_path=checkpoint_path,
        temperature=T,
        fit_info=fit_info,
        output_path=output_path,
    )

    # Save calibration record JSON
    record = CalibrationRecord(
        temperature=T,
        fit_info=fit_info,
        fitted_on_split="val",
        source_checkpoint=str(checkpoint_path),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_calibration(record, output_dir / "calibration_record.json")

    print(f"\n{'='*60}")
    print("AEGIS AUDIO TEMPERATURE SCALING — DONE")
    print(f"{'='*60}")
    print(f"  Temperature T        : {T:.6f}")
    print(f"  NLL before           : {fit_info.get('nll_before_calibration', 'N/A')}")
    print(f"  NLL after            : {fit_info.get('nll_after_calibration', 'N/A')}")
    print(f"  NLL reduction        : {fit_info.get('nll_reduction', 'N/A')}")
    print(f"  Val samples          : {len(logits)}")
    print(f"  Calibrated checkpoint: {output_path}")
    print(f"  Calibration record   : {output_dir / 'calibration_record.json'}")
    print(f"{'='*60}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
