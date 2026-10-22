# Dataset Preparation & Split Isolation

To prevent data leakage:
- **Audio**: Splits are isolated by speaker ID (`speaker_id`).
- **Video**: Sequences from the same original video are confined to one split.
- **Image**: Unseen generator evaluations (e.g. A07-A19) are strictly held out.
