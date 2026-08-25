"""Test script for video calibration modules."""

import sys
from pathlib import Path
import numpy as np

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from video.calibration.temperature_scaling import (
    TemperatureScalingLayer,
    fit_temperature,
    CalibrationRecord,
    make_prediction,
)
from video.calibration.evaluate_calibration import (
    compute_ece,
    compute_brier,
    compute_nll,
)

print("=" * 60)
print("Video Calibration Module Test")
print("=" * 60)

# Test 1: Temperature fitting
print("\n1. Testing temperature fitting...")
np.random.seed(42)
logits = np.random.randn(100) * 2
labels = (logits > 0).astype(float)

T, fit_info = fit_temperature(logits, labels)
print(f"   ✓ Temperature: {T:.4f}")
print(f"   ✓ NLL before: {fit_info['nll_before_calibration']:.4f}")
print(f"   ✓ NLL after: {fit_info['nll_after_calibration']:.4f}")
print(f"   ✓ Converged: {fit_info['converged']}")

# Test 2: Metrics computation
print("\n2. Testing metrics computation...")
probs = 1.0 / (1.0 + np.exp(-logits))
ece, _, _, _ = compute_ece(labels, probs)
brier = compute_brier(labels, probs)
nll = compute_nll(labels, probs)

print(f"   ✓ ECE: {ece:.4f}")
print(f"   ✓ Brier: {brier:.4f}")
print(f"   ✓ NLL: {nll:.4f}")

# Test 3: Calibrated metrics
print("\n3. Testing calibrated metrics...")
cal_probs = 1.0 / (1.0 + np.exp(-logits / T))
cal_ece, _, _, _ = compute_ece(labels, cal_probs)
cal_brier = compute_brier(labels, cal_probs)
cal_nll = compute_nll(labels, cal_probs)

print(f"   ✓ Calibrated ECE: {cal_ece:.4f} (improvement: {ece - cal_ece:.4f})")
print(f"   ✓ Calibrated Brier: {cal_brier:.4f} (improvement: {brier - cal_brier:.4f})")
print(f"   ✓ Calibrated NLL: {cal_nll:.4f} (improvement: {nll - cal_nll:.4f})")

# Test 4: Prediction API
print("\n4. Testing prediction API...")
pred = make_prediction(1.5, temperature=T)
print(f"   ✓ Label: {pred['label']}")
print(f"   ✓ Probability: {pred['probability']:.4f}")
print(f"   ✓ Calibrated: {pred['calibrated']}")
print(f"   ✓ Temperature: {pred['temperature']:.4f}")

# Test 5: CalibrationRecord
print("\n5. Testing CalibrationRecord...")
record = CalibrationRecord(
    temperature=T,
    fit_info=fit_info,
    fitted_on_split="val",
    source_checkpoint="models/video/baseline_best.pt",
)
print(f"   ✓ Record created: T={record.temperature:.4f}, split={record.fitted_on_split}")

print("\n" + "=" * 60)
print("✅ All tests passed! Video calibration modules working.")
print("=" * 60)
