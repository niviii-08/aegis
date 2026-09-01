# AEGIS Explainability Limitations Documentation

**Date**: August 29, 2026  
**Status**: Honest assessment of explanation method limitations

## Overview

This document provides comprehensive documentation of the limitations of each explainability method implemented in AEGIS. Understanding these limitations is crucial for proper interpretation of explanations and avoiding overconfidence in model decisions.

---

## Image Explainability (Grad-CAM)

### Method: Gradient-weighted Class Activation Mapping (Grad-CAM)

**Reference**: Selvaraju et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." ICCV.

### How It Works

Grad-CAM uses the gradients flowing into the final convolutional layer to produce a coarse localization map highlighting important regions:

```
L_Grad-CAM = ReLU(Σ_k α_k^c A^k)
```

where α_k^c are the global average pooled gradients and A^k are the activation maps.

### Known Limitations

1. **Spatial Resolution Limitation**
   - Heatmap resolution is limited by the target convolutional layer
   - For EfficientNet-B4, final conv features are much smaller than input (224×224)
   - Fine-grained details may be missed in the explanation

2. **Gradient-Based Instability**
   - Gradient-based methods can be noisy and unstable
   - Sensitive to small changes in input
   - May produce inconsistent explanations for similar inputs

3. **Single-Frame Limitation**
   - Only explains single static images
   - No temporal context (not applicable for static images, but relevant for video)
   - Cannot capture sequential dependencies

4. **Model Decision Correspondence**
   - Explanation corresponds to the model's decision boundary, not ground truth
   - If the model is wrong, the explanation may still highlight "important" regions
   - Does not indicate whether the model is correct

5. **Generalization Issues**
   - May not generalize well to unseen generators
   - Features that the model focuses on for seen generators may not transfer
   - Cannot detect distribution shift

6. **Class Bias**
   - Explanations may reflect training data biases
   - If training data has systematic artifacts, explanations may highlight those
   - Does not detect dataset biases

7. **Interpretation Ambiguity**
   - Heatmap intensity does not directly correlate with feature importance
   - High activation regions may be artifacts rather than discriminative features
   - Requires domain expertise to interpret correctly

### Reliability Score: 0.7/1.0

**Rationale**: Grad-CAM is a well-established method with theoretical grounding, but has significant limitations in resolution, stability, and generalization.

---

## Audio Explainability (Saliency Maps)

### Method: Gradient-based Input Saliency

**Reference**: Simonyan et al. (2013). "Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps." ICLR workshop.

### How It Works

Saliency maps compute the absolute gradient of the prediction score with respect to the input:

```
S = |∇P(fake)|
```

For mel-spectrograms, this produces a 2-D saliency map overlaying time-frequency information. For wav2vec2 features, this produces per-token saliency.

### Known Limitations

1. **Temporal Resolution Limitation**
   - Saliency is computed on fixed temporal windows
   - May miss longer temporal dependencies in audio
   - Time-frequency trade-off affects explanation quality

2. **Feature Space Limitation**
   - Saliency is computed in feature space, not raw audio
   - May not directly correspond to interpretable audio features
   - Requires understanding of mel-frequency or wav2vec2 feature spaces

3. **Gradient Noise**
   - Audio gradients can be particularly noisy due to high-dimensional features
   - May produce fragmented saliency patterns
   - Requires smoothing for interpretability

4. **Frequency Interpretation**
   - Mel-frequency bins are not linearly related to perceived pitch
   - Difficult to interpret saliency in terms of audible features
   - Requires audio domain expertise

5. **Model Architecture Dependence**
   - Different architectures (mel_cnn vs wav2vec2_cnn) produce different saliency patterns
   - Cannot directly compare explanations across architectures
   - Explanation quality depends on feature representation

6. **No Ground Truth**
   - No established ground truth for audio deepfake explanations
   - Difficult to validate if explanations highlight truly discriminative features
   - May highlight spurious correlations

### Current Status: NOT AVAILABLE

**Reason**: Audio model not trained

**Implementation Status**: ✅ Code implemented, ❌ Model unavailable

### Reliability Score: 0.0/1.0 (when available)

**Rationale**: Audio saliency is theoretically sound but heavily dependent on feature quality and lacks validation mechanisms.

---

## Video Explainability (Temporal Attention)

### Method: Grad-CAM for Video Frames

**Reference**: Selvaraju et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." ICCV.

### How It Works

Grad-CAM is applied to individual frames in a video sequence, producing temporal sequences of spatial heatmaps. This can show which regions are important across time.

### Known Limitations

1. **Frame Independence**
   - Current implementation applies Grad-CAM to each frame independently
   - Does not capture temporal dependencies between frames
   - May miss motion patterns that are important for deepfake detection

2. **Temporal Sampling**
   - Limited to fixed number of frames (e.g., 16 frames)
   - May miss important temporal segments
   - Sampling rate affects explanation quality

3. **Computational Cost**
   - Computing Grad-CAM for each frame is expensive
   - May not scale to long videos
   - Requires trade-offs between explanation quality and performance

4. **Motion Artifacts**
   - Deepfake detection often relies on motion inconsistencies
   - Frame-wise Grad-CAM may not capture motion-based features
   - May focus on static facial features rather than temporal dynamics

5. **Video Complexity**
   - Videos have additional complexity (lighting, compression artifacts)
   - Explanations may highlight technical artifacts rather than deepfake features
   - Difficult to disentangle model reasoning from technical factors

6. **Representative Frame Selection**
   - Choosing which frames to explain is non-trivial
   - May miss the most discriminative frames
   - Requires additional logic for frame selection

### Current Status: NOT AVAILABLE

**Reason**: Video model not trained

**Implementation Status**: ✅ Code implemented, ❌ Model unavailable

### Reliability Score: 0.5/1.0 (when available)

**Rationale**: Video Grad-CAM extends image Grad-CAM but adds significant complexity around temporal reasoning and frame selection.

---

## Cross-Modal Comparison

### Method Comparison Table

| Modality | Method | Reliability | Resolution | Temporal | Validation | Current Status |
|----------|--------|-------------|------------|----------|------------|----------------|
| Image | Grad-CAM | 0.7 | Medium (layer-limited) | N/A | Limited | ✅ Working |
| Audio | Saliency | 0.0 (estimated) | High (feature-space) | High | None | ❌ No model |
| Video | Temporal Grad-CAM | 0.5 (estimated) | Medium (frame-limited) | Low | None | ❌ No model |

### Key Differences

1. **Spatial vs Temporal**
   - Image: Purely spatial explanation
   - Audio: Temporal-frequency explanation
   - Video: Spatiotemporal explanation (complex)

2. **Feature Space**
   - Image: Raw pixel space (more interpretable)
   - Audio: Feature space (less interpretable)
   - Video: Frame pixel space (moderately interpretable)

3. **Validation Difficulty**
   - Image: Easier to validate spatial features
   - Audio: Harder to validate audio features
   - Video: Hardest to validate spatiotemporal features

---

## General Limitations Across All Methods

### 1. Model Decision Correspondence

**Issue**: All explanations correspond to model decisions, not ground truth.

**Impact**: If the model is incorrect, the explanation may still be "internally consistent" but misleading.

**Mitigation**: Always cross-reference explanations with known ground truth when available.

### 2. Training Data Bias

**Issue**: Explanations may reflect biases in training data rather than genuine features.

**Impact**: May highlight dataset artifacts rather than discriminative deepfake features.

**Mitigation**: Use diverse training data and analyze explanations across different data subsets.

### 3. Distribution Shift

**Issue**: Explanations may not generalize to unseen generators.

**Impact**: Features that explain seen generators may not transfer to novel ones.

**Mitigation**: Validate explanations on held-out generators when available.

### 4. Interpretation Subjectivity

**Issue**: Explanation interpretation requires domain expertise.

**Impact**: Different users may interpret the same explanation differently.

**Mitigation**: Provide clear guidelines and examples for explanation interpretation.

### 5. False Confidence

**Issue**: Visual explanations may create false confidence in model decisions.

**Impact**: Users may overtrust explanations that have significant limitations.

**Mitigation**: Always display reliability scores and limitations alongside explanations.

---

## Recommendations for Usage

### For Image Explanations

1. **Use as supplementary information** - never as primary decision criteria
2. **Focus on high-activation regions** - these are most likely to be meaningful
3. **Compare multiple samples** - look for consistent patterns
4. **Check for dataset artifacts** - ensure explanations aren't highlighting data issues
5. **Consider resolution limits** - fine details may be missed

### For Audio Explanations (when available)

1. **Understand feature space** - mel-frequency and wav2vec2 require different interpretation
2. **Look for temporal patterns** - focus on time segments with high saliency
3. **Cross-reference with spectrogram** - ensure saliency aligns with visible features
4. **Beware of noise** - audio gradients can be particularly noisy
5. **Validate with domain experts** - audio features require specialized knowledge

### For Video Explanations (when available)

1. **Consider temporal context** - look for patterns across frames
2. **Focus on motion patterns** - deepfakes often have temporal inconsistencies
3. **Check frame selection** - ensure representative frames are explained
4. **Be aware of artifacts** - compression and lighting can create spurious explanations
5. **Use as temporal guide** - explanations should inform temporal analysis

---

## Integration with Dashboard

### Current Status

**Image**: ✅ Ready for dashboard integration
- API provides standardized explanation format
- Includes prediction probability and class
- Documents limitations clearly
- Generates visualizations automatically

**Audio**: ❌ Not ready for dashboard integration
- Model not trained
- Cannot generate real explanations
- Infrastructure ready but blocked by data pipeline

**Video**: ❌ Not ready for dashboard integration
- Model not trained
- Cannot generate real explanations
- Infrastructure ready but blocked by data pipeline

### Integration Requirements

1. **Display prediction information first**
   - Always show predicted class and probability
   - Explanations are secondary information

2. **Include reliability scores**
   - Display reliability score prominently
   - Low reliability should trigger caution

3. **Show limitations**
   - Display relevant limitations for each modality
   - Provide tooltips or expandable sections

4. **Avoid overconfidence**
   - Use cautious language in UI
   - Don't present explanations as definitive

5. **Provide context**
   - Include method information (e.g., "Grad-CAM")
   - Show explanation generation timestamp

---

## Future Improvements

### Short-term (when models available)

1. **Train audio and video models** - enable multi-modal explanations
2. **Add captum library** - improve Grad-CAM implementation
3. **Implement LIME** - provide complementary local explanations
4. **Add SHAP values** - provide feature importance scores
5. **Create explanation validation** - user studies to assess quality

### Long-term (research directions)

1. **Develop temporal explanation methods** - specifically for video
2. **Implement attention visualization** - show internal model attention
3. **Add counterfactual explanations** - "what would change this prediction?"
4. **Develop concept-based explanations** - explain in terms of deepfake artifacts
5. **Create explanation benchmarks** - establish quality metrics

---

## Scientific Integrity Statement

**Principles**:
1. ✅ Explanations correspond to actual model predictions
2. ✅ Prediction probabilities and classes are displayed
3. ✅ Limitations are clearly documented
4. ✅ No explanations are fabricated
5. ✅ Explanations are verified with real model inference
6. ✅ Real dataset samples are used for example outputs

**Current Blockers**:
1. ❌ Only image model available for real explanations
2. ❌ Audio and video models not trained
3. ❌ Limited validation of explanation quality
4. ❌ No user studies to assess interpretability
5. ❌ Cannot assess generalization to unseen generators

**Status**: Image explainability infrastructure is functional and scientifically sound, but full multi-modal explainability requires resolving the same data pipeline issues that block the entire AEGIS project.

---

**Document Updated**: August 29, 2026  
**Next Review**: After audio/video model training  
**Contact**: For questions about explanation limitations, refer to individual method documentation and this comprehensive assessment.