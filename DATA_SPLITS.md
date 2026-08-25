# AEGIS Image Data Splits

This document describes the **generalization-aware** image split system used by AEGIS.
It is not a conventional random train/validation/test partition. The objective is to
support evaluation of **seen forgery generators** versus **unseen forgery generators**
while preventing several forms of leakage that invalidate generalization claims.

Configuration lives in `configs/image_split.yaml`. Split CSVs are written to
`data/processed/image/splits/`. Machine-readable statistics are in
`reports/split_statistics.json`.

Rebuild splits:

```bash
python -m image.splits.generator_split
python -m image.splits.validate_splits
```

---

## Why random image splitting is insufficient

Randomly shuffling individual images into train/validation/test fails for deepfake
research because it ignores the **generative process** that produced each sample.

If images from the same person, the same upstream source photograph, the same
StyleGAN latent, or the same forgery pipeline appear in both training and test,
reported accuracy mixes **memorization** with **generalization**. A model can appear
strong simply because it recognizes dataset-specific artifacts, identities, or
generator fingerprints already seen during training.

AEGIS therefore assigns samples to evaluation roles using:

1. **Generator policy** — which forgery methods may appear in each split.
2. **Proxy identity keys** — derived from upstream sample identifiers.
3. **Content hashes** — SHA-256 of on-disk JPEG bytes.
4. **Source-image keys** — normalized upstream provenance paths when available.

---

## Split roles

| Role | Purpose |
|------|---------|
| `train` | Model fitting on seen generators only |
| `val` | Hyperparameter and checkpoint selection on seen generators |
| `test_seen` | Held-out evaluation on generators present during training |
| `test_unseen` | Held-out evaluation on forgery generators never seen in train/val |

---

## Seen generators

Configured in `configs/image_split.yaml` under `split_generators`.

| Generator | Description | Present locally |
|-----------|-------------|-----------------|
| `ffhq_authentic` | Unmodified FFHQ real faces | Yes (70,000) |
| `stylegan` | StyleGAN faces from 1-Million-Fake-Faces | Yes (70,000) |

Both generators appear in `train`, `val`, and `test_seen` for the current corpus.

---

## Unseen generators

Unseen forgery generators are **fixed in config** and are never chosen randomly at
runtime. The current policy reserves:

- `deepfakes`
- `face2face`
- `faceswap`
- `neuraltextures`
- `faceshifter`
- `deepfake_detection`

These correspond to FaceForensics++ manipulation families documented under
`data/raw/image/FaceForensics-master/`, but **no manipulated media from these
generators is present locally yet**.

When unseen-generator media is ingested into the manifest, samples whose normalized
generator matches the unseen list are routed exclusively to `test_unseen.csv`.

---

## Exact sample counts (current corpus)

Built from `data/processed/image/manifest.csv` on 2026-08-15.

| Split role | Total | Real | Fake |
|------------|------:|-----:|-----:|
| `train` | 100,000 | 50,000 | 50,000 |
| `val` | 20,000 | 10,000 | 10,000 |
| `test_seen` | 20,000 | 10,000 | 10,000 |
| `test_unseen` | 0 | 0 | 0 |
| **Total assigned** | **140,000** | **70,000** | **70,000** |

Upstream mapping while unseen media is absent:

| Manifest split | AEGIS role |
|----------------|------------|
| `train` | `train` |
| `valid` | `val` |
| `test` | `test_seen` |

---

## Class balance

| Split role | Real fraction | Fake fraction | Minority fraction |
|------------|--------------:|--------------:|------------------:|
| `train` | 0.500 | 0.500 | 0.500 |
| `val` | 0.500 | 0.500 | 0.500 |
| `test_seen` | 0.500 | 0.500 | 0.500 |
| `test_unseen` | n/a (empty) | n/a (empty) | n/a (empty) |

Validation fails if any **non-empty** split drops below a 10% minority-class fraction.

---

## How identity leakage is prevented

The manifest marks `identity_id=unknown` because the upstream CSV `id` field is a
**per-image sample identifier**, not verified person identity. AEGIS still derives a
deterministic **proxy identity key** to block cross-split reuse of the same upstream
sample:

| Label | Identity key format | Example |
|-------|---------------------|---------|
| Real | `ffhq_source:{id}` | `ffhq_source:31355` |
| Fake | `stylegan_face:{id}` | `stylegan_face:FZV5C5L0AI` |

`src/image/splits/leakage_checker.py` asserts zero overlap of identity keys between
incompatible split pairs (train vs test_seen, train vs test_unseen, val vs test splits,
test_seen vs test_unseen).

The upstream real-vs-fake CSV splits already have zero cross-split ID overlap; the AEGIS
split preserves that property.

---

## How generator leakage is prevented

1. **Policy enforcement** — only generators listed under `split_generators.train` may
   appear in `train.csv`.
2. **Unseen holdout** — generators listed under `generator_taxonomy.unseen_forgery`
   must never appear in training. Validation raises a hard error if they do.
3. **Deterministic routing** — when unseen media exists, those samples are assigned to
   `test_unseen` regardless of upstream folder names.

Currently, training contains only `ffhq_authentic` and `stylegan`. No unseen generator
samples exist locally, so generator generalization cannot yet be measured empirically.

---

## Near-duplicate and source-image leakage

| Check | Mechanism |
|-------|-----------|
| Near-duplicate leakage | Shared SHA-256 `file_hash` across incompatible splits |
| Source-image leakage | Shared normalized `original_source` path as `source_image_key` |

The manifest audit found **zero duplicate content hashes** across the 140k-image corpus.

---

## Limitations (important)

A scientifically valid **unseen-generator** benchmark is **not yet possible** with the
local corpus alone.

### Missing metadata and media

| Gap | Impact |
|-----|--------|
| `multi_generator_forgery_media` | Only StyleGAN fakes exist; no DeepFakes/Face2Face/etc. |
| `faceforensics_plus_downloaded_sequences` | FF++ code and split JSON exist, but videos/frames are absent |
| `verified_cross_sample_person_identity_metadata` | Cannot enforce person-level disjointness beyond proxy sample IDs |

### What the current split can and cannot measure

| Evaluation | Supported now? |
|------------|----------------|
| Detection on seen StyleGAN fakes vs FFHQ reals | Yes (`test_seen`) |
| Generalization to unseen forgery generators | **No** (`test_unseen` is empty) |
| Person-level identity generalization | Partial (proxy keys only) |

### Architecture readiness

When additional datasets are added:

1. Extend `data/processed/image/manifest.csv` via `python -m image.data.manifest_builder`.
2. Ensure `generator` and `original_source` are populated for new forgeries.
3. Re-run `python -m image.splits.generator_split`.

Samples matching configured unseen generators will populate `test_unseen.csv` automatically
without changing the split code path.

---

## Validation

`src/image/splits/validate_splits.py` fails loudly on:

- Shared identity keys across incompatible splits
- Shared file hashes across incompatible splits
- Shared source-image keys across incompatible splits
- Unseen generators appearing in training
- Catastrophic class imbalance (< 10% minority class)

See `tests/test_leakage_checker.py` for unit coverage of each failure mode.
