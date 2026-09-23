# Forgetting-Risk Prediction ML

End-to-end tabular ML pipeline embedded in AEGIS: predicts **forgetting risk** after a
learning/review event using behavioral features (interval since last review, rolling accuracy,
item difficulty, circadian cues, modality tag).

## Design choices (recruiter-friendly)

| Topic | Implementation |
| --- | --- |
| Reproducibility | Fixed seeds, versioned dataset manifest, copied training config in artifacts |
| Splits | **Time-aware** 70/15/15 on `event_timestamp` (no random shuffle) |
| Leakage | Deny-list for IDs, timestamps, and future outcomes |
| Models | Logistic regression baseline + **XGBoost** primary |
| Imbalance | `class_weight=balanced` (LR), `scale_pos_weight` (XGBoost) |
| CV | **TimeSeriesSplit** on training window only |
| Calibration | Isotonic (default) or sigmoid on validation fold |
| Explainability | Gain-based feature importance + optional SHAP |
| Inference | Loads `forgetting_model.joblib` — **never retrains** on predict |

## Train

From repository root:

```bash
python -m ml.training.train
```

Quick CI/test config:

```bash
python -m ml.training.train --config ml/config/quick.yaml
```

## Artifacts (`ml/artifacts/forgotting_risk/`)

- `forgetting_model.joblib` — calibrated estimator + metadata
- `metrics.json` — validation/test + CV + baseline comparison
- `feature_metadata.json` — schema/version
- `feature_importance.json`
- `shap_global_importance.json` / `shap_individual_example.json` (when SHAP enabled)
- `performance_report.md`

## Which metric matters?

**PR-AUC (average precision)** and **recall at a sensible threshold** matter most: false
negatives (learner flagged as low-risk who then forgets) drive poor learning outcomes. Accuracy
can look high when most events are low-risk negatives.

## API

`POST /predict/forgetting` accepts JSON feature payloads matching `feature_metadata.json`.
