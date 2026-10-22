# Video Temporal Modeling Guide

Video manipulation detection uses:
- EfficientNet-B4 frame feature extraction (1792 dimensions)
- 2-layer Bidirectional LSTM temporal sequence model
- Uniform frame stride sampling (10-16 frames per sequence)
