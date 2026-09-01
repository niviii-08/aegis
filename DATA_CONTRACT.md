# AEGIS Data Contract

## 1. Central Sources of Truth

*   **Raw Data Manifest (`data/processed/<modality>/manifest.csv`)**:
    *   Defines the complete inventory of all known raw files.
    *   Never modified during model training.
*   **Sample Registry (`data/processed/<modality>/sample_registry.csv`)**:
    *   The single authoritative source for pipeline state.
    *   Enforces that every row marked `PROCESSED` has exactly one valid model-consumable artifact.
    *   Splits are ONLY generated from PROCESSED registry entries.
*   **Split Manifests (`data/processed/<modality>/splits/*.csv`)**:
    *   Defines exactly which samples belong to `train`, `val`, `test_seen`, and `test_unseen`.
    *   Derived purely from the sample registry.

## 2. Sample Registry Status Values

Every entry in the registry must have one of these valid statuses:

*   **RAW**: Sample exists in manifest but has not been successfully processed.
*   **PROCESSED**: Sample has been processed, crop exists, npy exists, and metadata is verified.
*   **FAILED**: Preprocessing attempted but failed (e.g. no face found, corrupted image).
*   **EXCLUDED**: Sample manually excluded from the pipeline.

## 3. Split Rules

*   A sample must NEVER appear in the training split if its processed artifact does not exist (i.e. if it is not `PROCESSED`).
*   Splits are disjoint by `sample_id`.
*   The `test_unseen` split must not share any generators with the `train` split.
*   If no unseen-generator data is available, `test_unseen.csv` must remain empty (0 rows).

## 4. Preprocessing Guarantees

*   **Idempotency**: Running the pipeline multiple times on the same input data must yield the same outputs without creating duplicates.
*   **Hashing**: Every `PROCESSED` sample must have a valid SHA-256 hash representing the exact contents of its processed artifact.
*   **Versioning**: Every `PROCESSED` sample is tagged with a `preprocessing_version`, `code_version`, and `configuration_hash`.
*   **Crash Recovery**: If preprocessing is interrupted, a restart automatically recovers `PROCESSING` entries based on artifact existence.
*   **Atomic Updates**: Registry and metadata are flushed atomically via temp-file + rename.
*   **Artifact Verification**: A sample is only marked `PROCESSED` after both crop and `.npy` are verified to exist and be readable.

## 5. CLI Interface

The preprocessing pipeline supports:

```bash
# Dry-run (no modifications)
python -m src.image.preprocessing.preprocess --dry-run

# Debug mode with limit
python -m src.image.preprocessing.preprocess --limit 100

# Full run with batch checkpointing
python -m src.image.preprocessing.preprocess --batch-size 10000

# Retry failed samples
python -m src.image.preprocessing.preprocess --retry-failures
```

## 6. Current State (Image Modality)

*   Raw Dataset Size: 140,000
*   Processed Dataset Size: 284 (237 original + 47 from validation trials)
*   Failed: 5
*   Remaining RAW: 139,711
*   Infrastructure: Verified and Idempotent
*   Research Readiness: Incomplete (pending full preprocessing)
