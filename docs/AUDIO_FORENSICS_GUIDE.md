# Audio Forensics Guide

AEGIS uses 128-band Mel-Spectrogram features computed over 16 kHz audio:
- **STFT window**: 512 samples
- **Hop length**: 160 samples (10 ms frame step)
- **Target frame length**: 600 frames (6 seconds)
- **Data Augmentation**: TimeMasking and FrequencyMasking (SpecAugment)
