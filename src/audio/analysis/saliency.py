"""Gradient-based input saliency maps for AEGIS audio classifiers.

Theory
------
Saliency via vanilla input gradients (Simonyan et al., 2013) answers:
"Which *time/frequency* positions in the input contribute most to the
model's P(fake) decision?"

For a model that maps features ``x`` - logit ``z`` - P(fake) = sigmoid(z),
the saliency map ``S`` is the absolute gradient of the prediction score w.r.t.
the input.  Two flavours are produced depending on model architecture:

* **mel_cnn (A5)** -- input is a 2-D mel-spectrogram
  ``[1, n_mels, T]``.  Saliency is a 2-D map of equal size, overlayed on
  the spectrogram with matplotlib's ``hot`` colormap.  Axes are:
  ``x`` = time (frames, 10 ms/frame), ``y`` = mel frequency bin (Hz scale).

* **wav2vec2_cnn (A5)** -- input is a 1-D feature sequence
  ``[T, 768]`` (or similar hidden dim).  A per-token saliency is obtained
  by reducing the gradient magnitude across the channel axis:
  ``S[t] = mean_d |d P(fake) / d x[t, d]|``.  Plotted as a 1-D bar chart
  with ``x`` = sequence time (ms via wav2vec2's 20 ms hop) and ``y`` =
  saliency magnitude.

The ``saliency_for_api(features, model)`` helper renders a small RGB image
suitable for embedding in the FastAPI JSON response (A10-style endpoint).

References
----------
Simonyan, K. et al. (2013). Deep Inside Convolutional Networks: Visualising
Image Classification Models and Saliency Maps. ICLR workshop.

Usage (CLI)::

    python -m audio.analysis.saliency \\
        --checkpoint  models/audio/baseline_best.pt \\
        --input       path/to/audio.wav

The CLI also supports ``--samples`` which walks the ``test_seen`` split
and writes one plot per category (real / fake / false-positive) into
``reports/audio/saliency_samples/``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# - bootstrap src/ on sys.path -
_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from audio.training.dataset import (
    AudioDataset,
    AudioSampleRecord,
    resolve_samples,
)
from audio.training.evaluate import load_checkpoint_model
from audio.training.utils import (
    find_project_root,
    load_training_config,
    resolve_device,
    set_seed,
)
from audio.preprocessing.preprocess import (
    extract_mel_spectrogram,
    extract_wav2vec2_features,
    load_preprocess_config,
    load_wav2vec2_model,
    PreprocessConfig,
)

logger = logging.getLogger(__name__)


# -
# Gradient / saliency core
# -

def _detect_feature_mode(model: nn.Module) -> str:
    """Return ``"mel_spectrogram"`` or ``"wav2vec2"`` based on ``model.describe()``."""
    if hasattr(model, "describe"):
        mt = model.describe().get("model_type", "")
        if mt == "mel_cnn":
            return "mel_spectrogram"
        if mt == "wav2vec2_cnn":
            return "wav2vec2"
    # Fallback heuristic from first layer shape
    for name, p in model.named_parameters():
        if name.startswith("conv_blocks"):
            return "mel_spectrogram"
        if name.startswith("conv_layers"):
            return "wav2vec2"
    return "mel_spectrogram"


@torch.enable_grad()
def compute_saliency_map(
    features_tensor: torch.Tensor,
    model: nn.Module,
    *,
    target_class: int | None = None,
    reduce_channels: bool = True,
) -> tuple[np.ndarray, float, float]:
    """Compute gradient-based input saliency for one sample.

    Parameters
    ----------
    features_tensor : torch.Tensor
        Model input tensor with a batch dimension.
        * mel_spectrogram: ``[1, 1, n_mels, T]``
        * wav2vec2: ``[1, T, D]``
    model : nn.Module
        Trained audio classifier (``AudioMelModel`` or ``AudioBaselineModel``).
    target_class : int or None
        Class index w.r.t. which the gradient is taken.  If ``None``, uses the
        predicted class (argmax or threshold 0.5 on P(fake)).  Because the model
        is binary, this almost always equals the predicted P(fake) class.
    reduce_channels : bool
        For wav2vec2 ``[1, T, D]`` inputs, reduce the gradient over the
        feature dimension ``D`` via mean abs to get a per-token ``[T]`` map.

    Returns
    -------
    saliency : np.ndarray
        Normalised saliency map (values in ``[0, 1]``).  Same *spatial* shape
        as the input minus batch / channel dims:
        * mel_spectrogram: ``[n_mels, T]``
        * wav2vec2 (reduced): ``[T]``
        * wav2vec2 (full): ``[T, D]``
    p_fake : float
        Model output P(fake) for the sample (diagnostic).
    logit : float
        Raw model logit for the sample (diagnostic).
    """
    model.eval()
    x = features_tensor.detach().clone().requires_grad_(True)
    device = next(model.parameters()).device
    x = x.to(device)

    logits = model(x)                        # [B]
    p = torch.sigmoid(logits)                 # [B]

    if target_class is None:
        target_class = 1 if p.item() >= 0.5 else 0

    # Binary gradient target: score for class 1 (fake) or 1-score for class 0
    if target_class == 1:
        score = logits.sum()
    else:
        score = (-logits).sum()

    if x.grad is not None:
        x.grad.zero_()
    score.backward()

    grad = x.grad.detach().cpu().numpy()      # [1, 1, n_mels, T] or [1, T, D]
    x.detach_()

    # Saliency = |gradient|, normalised to [0,1]
    if grad.ndim == 4:
        # mel_spectrogram: [1, 1, n_mels, T] -> [n_mels, T]
        raw = np.abs(grad[0, 0])
    elif grad.ndim == 3:
        # wav2vec2: [1, T, D] -> either [T, D] or [T]
        if reduce_channels:
            raw = np.abs(grad[0]).mean(axis=-1)   # [T]
        else:
            raw = np.abs(grad[0])                  # [T, D]
    else:
        raise ValueError(f"Unexpected gradient shape {grad.shape}")

    saliency = _normalise(raw)
    return saliency, float(p.item()), float(logits.item())


def _normalise(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1] (safe for near-zero arrays)."""
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


# -
# Feature extraction helpers  (wraps audio.preprocessing pipeline)
# -

@dataclass
class PreparedSample:
    """Everything needed to run + visualise saliency for one clip."""
    features_tensor: torch.Tensor      # [1, ...] ready for model input
    features_np: np.ndarray            # raw features for display (no batch dim)
    feature_mode: str                  # "mel_spectrogram" or "wav2vec2"
    true_label: int                    # 0=real, 1=fake
    clip_id: str
    source_path: Path | None


def prepare_features_from_file(
    audio_path: Path,
    *,
    prep_config: PreprocessConfig,
    device: torch.device,
    w2v_model: Any = None,
    w2v_processor: Any = None,
    true_label: int | None = None,
    clip_id: str | None = None,
) -> PreparedSample:
    """Extract model-ready features from a raw audio file.

    Returns a ``PreparedSample`` with ``features_tensor`` shaped for the
    configured ``prep_config.mode`` and cropped/padded consistently with the
    training ``AudioDataset`` preprocessing.
    """
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        import librosa
    except ImportError as exc:
        raise ImportError("librosa is required for audio-file saliency.") from exc

    waveform, _sr = librosa.load(str(audio_path), sr=prep_config.sample_rate, mono=True)

    mode = prep_config.mode
    if mode == "mel_spectrogram":
        mel = extract_mel_spectrogram(waveform, prep_config)     # [n_mels, T]
        features_np = mel
        tensor = _tensor_from_mel(mel, n_mels=prep_config.n_mels)
    else:
        if w2v_model is None or w2v_processor is None:
            raise RuntimeError(
                "wav2vec2 mode requires w2v_model + w2v_processor to be loaded."
            )
        emb = extract_wav2vec2_features(waveform, prep_config, w2v_model, w2v_processor)
        features_np = emb
        tensor = _tensor_from_w2v2(emb)

    return PreparedSample(
        features_tensor=tensor.unsqueeze(0).to(device),
        features_np=features_np,
        feature_mode=mode,
        true_label=true_label if true_label is not None else -1,
        clip_id=clip_id or audio_path.stem,
        source_path=audio_path,
    )


def prepare_features_from_record(
    sample: AudioSampleRecord,
    *,
    feature_mode: str,
    device: torch.device,
) -> PreparedSample:
    """Load preprocessed features from an ``AudioSampleRecord`` on disk."""
    features_np = np.load(sample.feature_path).astype(np.float32)

    if feature_mode == "mel_spectrogram":
        tensor = _tensor_from_mel(features_np, n_mels=features_np.shape[0])
    else:
        tensor = _tensor_from_w2v2(features_np)

    return PreparedSample(
        features_tensor=tensor.unsqueeze(0).to(device),
        features_np=features_np,
        feature_mode=feature_mode,
        true_label=sample.label,
        clip_id=sample.clip_id,
        source_path=None,
    )


def _tensor_from_mel(mel: np.ndarray, *, n_mels: int, target_T: int = 600) -> torch.Tensor:
    """Crop/pad a raw mel-spectrogram ``[n_mels, T]`` -> ``[1, n_mels, target_T]`` (centre-crop eval mode)."""
    if mel.ndim != 2:
        raise ValueError(f"Expected [n_mels, T] mel, got shape {mel.shape}")
    nm, T = mel.shape
    if T > target_T:
        start = (T - target_T) // 2
        mel = mel[:, start:start + target_T]
    elif T < target_T:
        pad = target_T - T
        pad_l = pad // 2
        pad_r = pad - pad_l
        mel = np.pad(mel, ((0, 0), (pad_l, pad_r)), mode="constant", constant_values=0)
    if nm != n_mels:
        raise ValueError(f"Mel bins mismatch: got {nm}, expected {n_mels}")
    return torch.from_numpy(mel.astype(np.float32)).unsqueeze(0)  # [1, n_mels, T]


def _tensor_from_w2v2(emb: np.ndarray, *, target_T: int = 600) -> torch.Tensor:
    """Pad/truncate wav2vec2 embedding ``[T, D]`` -> ``[target_T, D]``."""
    if emb.ndim != 2:
        raise ValueError(f"Expected [T, D] wav2vec2 embedding, got shape {emb.shape}")
    T, D = emb.shape
    if T > target_T:
        emb = emb[:target_T, :]
    elif T < target_T:
        pad = target_T - T
        emb = np.pad(emb, ((0, pad), (0, 0)), mode="constant", constant_values=0)
    return torch.from_numpy(emb.astype(np.float32))  # [T, D]


# -
# Visualisation
# -

def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401 -- re-exported below
    return plt


def plot_mel_saliency(
    mel_db: np.ndarray,
    saliency: np.ndarray,
    *,
    sample_rate: int = 16000,
    hop_length_ms: float = 10.0,
    n_mels: int = 128,
    title: str = "",
    p_fake: float | None = None,
    true_label: int | None = None,
) -> Any:
    """Overlay saliency map on a mel-spectrogram; returns matplotlib Figure.

    Axes:
        x = time (seconds)
        y = mel frequency bin index (labeled with approximate Hz at 4 points)
    """
    plt = _setup_matplotlib()

    # Time axis in seconds
    T = mel_db.shape[1]
    hop_s = hop_length_ms / 1000.0
    time_axis = np.arange(T) * hop_s

    # Mel-frequency tick labels (approximate -- we don't have the full mel filterbank)
    mel_ticks = np.linspace(0, n_mels - 1, 5).astype(int)
    # Map mel-bin index -> approximate Hz using librosa's mel-to-hz formula (linear approx for labels)
    try:
        import librosa
        hz_labels = [f"{int(librosa.mel_to_hz(m, sr=sample_rate, n_mels=n_mels))}" for _ in mel_ticks]
    except Exception:
        hz_labels = [str(int(i * sample_rate / 2 / (n_mels - 1))) for i in mel_ticks]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 1. Raw mel spectrogram (dB)
    ax = axes[0]
    mel_norm = _normalise(mel_db)
    ax.imshow(mel_norm, origin="lower", aspect="auto", cmap="magma",
              extent=[0, T * hop_s, 0, n_mels])
    ax.set_yticks(mel_ticks)
    ax.set_yticklabels(hz_labels)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel bin (approx Hz)")
    ax.set_title("Mel-spectrogram (input)")

    # 2. Saliency map alone
    ax = axes[1]
    ax.imshow(saliency, origin="lower", aspect="auto", cmap="hot",
              extent=[0, T * hop_s, 0, n_mels])
    ax.set_yticks(mel_ticks)
    ax.set_yticklabels(hz_labels)
    ax.set_xlabel("Time (s)")
    ax.set_title("Input-gradient saliency (|∇P(fake)|)")

    # 3. Overlay: mel (greys alpha) + saliency (hot additive)
    ax = axes[2]
    ax.imshow(mel_norm, origin="lower", aspect="auto", cmap="gray", alpha=0.55,
              extent=[0, T * hop_s, 0, n_mels])
    im = ax.imshow(saliency, origin="lower", aspect="auto", cmap="hot", alpha=0.7,
                   extent=[0, T * hop_s, 0, n_mels])
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04, label="saliency")
    ax.set_yticks(mel_ticks)
    ax.set_yticklabels(hz_labels)
    ax.set_xlabel("Time (s)")
    suffix = ""
    if p_fake is not None:
        suffix += f"  P(fake)={p_fake:.3f}"
    if true_label is not None and true_label >= 0:
        suffix += f"  y={'fake' if true_label == 1 else 'real'}"
    ax.set_title("Overlay (mel + saliency)" + suffix)

    fig.suptitle(title or "Gradient Saliency -- mel_cnn", fontsize=13)
    fig.tight_layout()
    return fig


def plot_w2v2_saliency(
    embedding: np.ndarray,
    saliency_1d: np.ndarray,
    *,
    token_ms: float = 20.0,
    title: str = "",
    p_fake: float | None = None,
    true_label: int | None = None,
) -> Any:
    """1-D bar-chart saliency over wav2vec2 token sequence + embedding heatmap."""
    plt = _setup_matplotlib()

    T = embedding.shape[0]
    token_s = token_ms / 1000.0
    time_axis = np.arange(T) * token_s

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    # 1. Embedding heatmap (transposed: channels on y)
    ax = axes[0]
    emb_norm = _normalise(embedding.T)
    ax.imshow(emb_norm, origin="lower", aspect="auto", cmap="viridis",
              extent=[0, T * token_s, 0, embedding.shape[1]])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("wav2vec2 feature dim")
    ax.set_title("wav2vec2 input embedding (T x D)")

    # 2. Per-token saliency bar chart
    ax = axes[1]
    bars = ax.bar(time_axis, saliency_1d, width=token_s * 0.9, color="#E53935",
                  edgecolor="black", linewidth=0.2, align="edge")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean |∇P(fake)| (normalised)")
    subtitle = "Saliency per wav2vec2 token"
    if p_fake is not None:
        subtitle += f"  P(fake)={p_fake:.3f}"
    if true_label is not None and true_label >= 0:
        subtitle += f"  y={'fake' if true_label == 1 else 'real'}"
    ax.set_title(subtitle)
    ax.set_xlim(0, T * token_s)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle(title or "Gradient Saliency -- wav2vec2_cnn", fontsize=13)
    fig.tight_layout()
    return fig


# -
# FastAPI endpoint helper
# -

def saliency_for_api(
    features: torch.Tensor | np.ndarray,
    model: nn.Module,
    *,
    device: torch.device | None = None,
    width_px: int = 480,
    height_px: int = 240,
) -> np.ndarray:
    """Render a compact RGB saliency image suitable for JSON embedding.

    Parameters
    ----------
    features : torch.Tensor or np.ndarray
        A single sample's input features:
        * mel: ``[n_mels, T]`` (numpy) or ``[1, n_mels, T]`` (tensor) or with leading batch.
        * wav2vec2: ``[T, D]`` or ``[1, T, D]``.
    model : nn.Module
        Trained classifier.
    device : torch.device, optional
        Inference device (auto-detected if ``None``).
    width_px, height_px : int
        Output image dimensions (the saliency viewport is rendered to this size).

    Returns
    -------
    np.ndarray, dtype=uint8, shape ``[height_px, width_px, 3]``
        RGB image of the saliency overlay, suitable for:
        ``base64.b64encode(img.tobytes())`` or direct ``PIL.Image.fromarray(img)``.
    """
    if device is None:
        device = next(model.parameters()).device

    plt = _setup_matplotlib()

    # Convert to tensor with leading batch dim
    if isinstance(features, np.ndarray):
        f_t = torch.from_numpy(features.astype(np.float32))
    else:
        f_t = features.detach().cpu()

    if f_t.ndim == 2:
        # Either [n_mels, T] (mel) or [T, D] (w2v2) -- heuristic
        if f_t.shape[0] < 200 and f_t.shape[1] > 500:
            mel = True
            inp = f_t.unsqueeze(0).unsqueeze(0).to(device)  # [1,1,n_mels,T]
        else:
            mel = False
            inp = f_t.unsqueeze(0).to(device)              # [1,T,D]
    elif f_t.ndim == 3:
        # [1, n_mels, T] or [1, T, D] or [B, T, D]
        if f_t.shape[-2] < 200 and f_t.shape[-1] > 500 and f_t.shape[0] <= 4:
            mel = True
            inp = f_t.unsqueeze(1).to(device) if f_t.shape[0] <= 4 else f_t.unsqueeze(0).to(device)
            if inp.ndim == 3:
                inp = inp.unsqueeze(1).to(device)
        else:
            mel = False
            inp = f_t.unsqueeze(0).to(device) if f_t.shape[0] > 4 else f_t.to(device)
            if inp.ndim == 2:
                inp = inp.unsqueeze(0).to(device)
    elif f_t.ndim == 4:
        mel = True
        inp = f_t.to(device)
    else:
        raise ValueError(f"Unsupported features shape {tuple(f_t.shape)}")

    # Force batch=1 for api
    inp = inp[:1]
    saliency, p_fake, _logit = compute_saliency_map(inp, model, reduce_channels=True)

    # Render to the requested pixel size via matplotlib, then capture to buffer
    import io
    from PIL import Image

    fig, ax = plt.subplots(1, 1, figsize=(width_px / 100, height_px / 100), dpi=100)
    if mel:
        # Saliency shape [n_mels, T] -- ensure input features available for overlay
        raw = inp.detach().cpu().numpy()[0, 0]
        mel_n = _normalise(raw)
        ax.imshow(mel_n, origin="lower", aspect="auto", cmap="gray", alpha=0.5)
        ax.imshow(saliency, origin="lower", aspect="auto", cmap="hot", alpha=0.7)
    else:
        ax.bar(np.arange(len(saliency)), saliency, color="#E53935",
               edgecolor="none", width=1.0)
        ax.set_xlim(0, len(saliency))
        ax.set_ylim(0, 1)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB").resize((width_px, height_px))
    return np.array(img, dtype=np.uint8)


# -
# Sample plots (real / fake / false-positive)
# -

def _predict_one(model: nn.Module, prepared: PreparedSample, device: torch.device) -> tuple[float, int]:
    model.eval()
    with torch.no_grad():
        logits = model(prepared.features_tensor.to(device))
        p = float(torch.sigmoid(logits).cpu().item())
    return p, (1 if p >= 0.5 else 0)


def generate_sample_plots(
    model: nn.Module,
    train_config: Any,
    prep_config: PreprocessConfig,
    device: torch.device,
    output_dir: Path,
    *,
    split: str = "test_seen",
    max_per_category: int = 1,
    threshold: float = 0.5,
    w2v_model: Any = None,
    w2v_processor: Any = None,
) -> list[Path]:
    """Write one plot per category (real, fake, false-positive) to ``output_dir``.

    For consistency we resolve samples from the ``split`` split manifest using
    the same resolution logic as evaluation + training.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_csv = train_config.split_dir / f"{split}.csv"
    if not split_csv.is_file():
        logger.warning("Split CSV not found: %s -- skipping sample plots.", split_csv)
        return []

    samples = resolve_samples(
        split_csv,
        train_config.metadata_path,
        train_config.project_root,
        preprocessing_version=train_config.preprocessing_version,
        feature_mode=train_config.feature_mode,   # type: ignore[arg-type]
    )
    if not samples:
        logger.warning("No samples resolved for split=%s.", split)
        return []

    feature_mode = _detect_feature_mode(model)
    dataset = AudioDataset(samples, feature_mode=feature_mode, is_training=False)

    reals, fakes, fps = [], [], []
    for idx, (x, y) in enumerate(dataset):
        rec = samples[idx]
        inp = x.unsqueeze(0).to(device)
        with torch.no_grad():
            p = float(torch.sigmoid(model(inp)).cpu().item())
        pred = 1 if p >= threshold else 0
        y_int = int(y.item())
        if y_int == 0 and len(reals) < max_per_category:
            reals.append((rec, inp, p, pred, y_int, x.numpy()))
        elif y_int == 1 and len(fakes) < max_per_category:
            fakes.append((rec, inp, p, pred, y_int, x.numpy()))
        elif y_int == 0 and pred == 1 and len(fps) < max_per_category:
            fps.append((rec, inp, p, pred, y_int, x.numpy()))
        if len(reals) >= max_per_category and len(fakes) >= max_per_category and len(fps) >= max_per_category:
            break

    # If we didn't find any FP (model is too good), just pick the highest-P real as the FP example
    if not fps and reals:
        # Re-scan for the highest-P real sample as a stand-in for FP
        candidate = None
        for idx, (x, y) in enumerate(dataset):
            rec = samples[idx]
            y_int = int(y.item())
            if y_int != 0:
                continue
            inp = x.unsqueeze(0).to(device)
            with torch.no_grad():
                p = float(torch.sigmoid(model(inp)).cpu().item())
            if candidate is None or p > candidate[2]:
                candidate = (rec, inp, p, 1 if p >= threshold else 0, y_int, x.numpy())
        if candidate is not None:
            fps.append(candidate)
            logger.warning("No true false-positives found; using highest-P real sample for the FP plot.")

    written: list[Path] = []
    for category, bucket in (("real", reals), ("fake", fakes), ("false_positive", fps)):
        if not bucket:
            logger.warning("No samples for category=%s -- skipping plot.", category)
            continue
        rec, inp, p, pred, y_int, features_raw = bucket[0]
        sal, p_sal, _ = compute_saliency_map(inp, model, reduce_channels=True)

        # Rebuild raw display features without batch/channel axes
        if feature_mode == "mel_spectrogram":
            if features_raw.ndim == 3:      # [1, n_mels, T]
                mel_display = features_raw[0]
            else:
                mel_display = features_raw
            fig = plot_mel_saliency(
                mel_db=mel_display,
                saliency=sal,
                sample_rate=prep_config.sample_rate,
                hop_length_ms=prep_config.hop_length_ms,
                n_mels=prep_config.n_mels,
                title=f"Saliency -- {category} clip  ({rec.clip_id[:32]})",
                p_fake=p_sal,
                true_label=y_int,
            )
        else:
            if features_raw.ndim == 3:       # w2v2 dataset returns [T, D] (no extra dim)
                emb_display = features_raw[0] if features_raw.shape[0] == 1 else features_raw
            else:
                emb_display = features_raw
            fig = plot_w2v2_saliency(
                embedding=emb_display,
                saliency_1d=sal,
                token_ms=20.0,
                title=f"Saliency -- {category} clip  ({rec.clip_id[:32]})",
                p_fake=p_sal,
                true_label=y_int,
            )

        safe_clip = rec.clip_id.replace(":", "__").replace("/", "_")
        out_path = output_dir / f"saliency_{category}_{safe_clip}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        import matplotlib.pyplot as plt_mod
        plt_mod.close(fig)
        logger.info("Saved saliency plot -> %s", out_path)
        written.append(out_path)

    return written


# -
# Single-file CLI plot
# -

def saliency_for_file(
    audio_path: Path,
    model: nn.Module,
    prep_config: PreprocessConfig,
    device: torch.device,
    *,
    output_dir: Path,
    w2v_model: Any = None,
    w2v_processor: Any = None,
    true_label: int | None = None,
) -> Path:
    """Run saliency for a single audio file and save the plot to ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_mode = _detect_feature_mode(model)
    # Force prep_config.mode to match model for display
    if prep_config.mode != feature_mode:
        prep_config = _override_mode(prep_config, feature_mode)

    prepared = prepare_features_from_file(
        audio_path,
        prep_config=prep_config,
        device=device,
        w2v_model=w2v_model,
        w2v_processor=w2v_processor,
        true_label=true_label,
    )
    sal, p_fake, _logit = compute_saliency_map(prepared.features_tensor, model, reduce_channels=True)

    if feature_mode == "mel_spectrogram":
        fig = plot_mel_saliency(
            mel_db=prepared.features_np,
            saliency=sal,
            sample_rate=prep_config.sample_rate,
            hop_length_ms=prep_config.hop_length_ms,
            n_mels=prep_config.n_mels,
            title=f"Saliency -- {audio_path.name}",
            p_fake=p_fake,
            true_label=prepared.true_label,
        )
    else:
        fig = plot_w2v2_saliency(
            embedding=prepared.features_np,
            saliency_1d=sal,
            token_ms=20.0,
            title=f"Saliency -- {audio_path.name}",
            p_fake=p_fake,
            true_label=prepared.true_label,
        )

    out = output_dir / f"saliency_{audio_path.stem}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt_mod
    plt_mod.close(fig)
    logger.info("Saved saliency plot -> %s  (P(fake)=%.3f)", out, p_fake)
    return out


def _override_mode(prep_config: PreprocessConfig, mode: str) -> PreprocessConfig:
    """Create a copy of ``prep_config`` with ``mode`` overridden (best-effort)."""
    from dataclasses import replace as _dc_replace
    try:
        return _dc_replace(prep_config, mode=mode)
    except Exception:
        return prep_config


# -
# CLI
# -

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AEGIS audio input-gradient saliency maps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Trained checkpoint (.pt), e.g. models/audio/baseline_best.pt.")
    p.add_argument("--input", type=str, default=None,
                   help="Path to a .wav/.flac file, OR 'samples' to emit real/fake/FP sample plots.")
    p.add_argument("--label", type=int, default=None, choices=(0, 1),
                   help="True label for a user-supplied --input file (0=real, 1=fake).")
    p.add_argument("--split", type=str, default="test_seen",
                   help="Split used for --input=samples (default: test_seen).")
    p.add_argument("--config", type=Path, default=None,
                   help="Path to audio_baseline.yaml (default: <root>/configs/audio_baseline.yaml).")
    p.add_argument("--prep-config", type=Path, default=None,
                   help="Path to audio_preprocessing.yaml.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Output directory for saliency PNGs (default: reports/audio/saliency_samples/).")
    p.add_argument("--project-root", type=Path, default=None,
                   help="AEGIS project root (auto-discovered if omitted).")
    p.add_argument("--seed", type=int, default=42, help="Deterministic seed.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)

    project_root = (args.project_root or find_project_root()).resolve()
    config_path = (args.config or project_root / "configs" / "audio_baseline.yaml").resolve()
    prep_config_path = (args.prep_config or project_root / "configs" / "audio_preprocessing.yaml").resolve()
    default_output = project_root / "reports" / "audio" / "saliency_samples"
    output_dir = (args.output_dir or default_output).resolve()

    set_seed(args.seed)
    device = resolve_device()
    logger.info("Device: %s", device)

    train_config = load_training_config(config_path, project_root)
    prep_config = load_preprocess_config(prep_config_path, project_root)

    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model, payload = load_checkpoint_model(checkpoint_path, device)
    model.eval()
    feature_mode = _detect_feature_mode(model)
    logger.info("Loaded model: %s  feature_mode=%s", model.describe().get("model_type", "?"), feature_mode)

    # Load wav2vec2 backbone lazily (only needed if we run from raw audio)
    w2v_model, w2v_processor = None, None
    if feature_mode == "wav2vec2" and args.input and args.input != "samples":
        logger.info("Loading wav2vec2 encoder for on-the-fly feature extraction...")
        w2v_model, w2v_processor = load_wav2vec2_model(prep_config)

    written: list[Path] = []

    if args.input is None or args.input.strip().lower() == "samples":
        if feature_mode == "wav2vec2":
            logger.info("Loading wav2vec2 encoder (for samples mode -- not strictly required because preprocessed features are loaded from disk).")
        written += generate_sample_plots(
            model, train_config, prep_config, device, output_dir,
            split=args.split, max_per_category=1, threshold=train_config.eval_threshold,
            w2v_model=w2v_model, w2v_processor=w2v_processor,
        )
    else:
        audio_path = Path(args.input).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(f"--input audio not found: {audio_path}")
        written.append(saliency_for_file(
            audio_path, model, prep_config, device,
            output_dir=output_dir,
            w2v_model=w2v_model, w2v_processor=w2v_processor,
            true_label=args.label,
        ))

    print("\n" + "=" * 60)
    print("AEGIS AUDIO SALIENCY -- DONE")
    print("=" * 60)
    print(f"  Model type  : {model.describe().get('model_type', '?')}")
    print(f"  Feature mode: {feature_mode}")
    print(f"  Output dir  : {output_dir}")
    for w in written:
        print(f"    [OK] {w.name}")
    if not written:
        print("  (no plots written -- split/samples may be empty)")
    print("=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
