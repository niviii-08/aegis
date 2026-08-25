"""Temperature Scaling for AEGIS video binary classifiers.

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


# ─────────────────────────────────────────────────────────────────────────────
# Prediction API helper
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
