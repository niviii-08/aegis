"""Real-time audio streaming inference and latency-vs-accuracy evaluation.

Usage::

    # Stream from a file
    python -m audio.training.streaming \\
        --checkpoint models/audio/baseline_best.pt \\
        --input path/to/audio.wav

    # Stream with temperature-scaled calibrated model
    python -m audio.training.streaming \\
        --checkpoint models/audio/baseline_best.pt \\
        --calibrated \\
        --input path/to/audio.wav

    # Stream from microphone (requires sounddevice)
    python -m audio.training.streaming \\
        --checkpoint models/audio/baseline_best.pt \\
        --input mic

    # Run latency-vs-accuracy parameter sweep over test_seen
    python -m audio.training.streaming \\
        --checkpoint models/audio/baseline_best.pt \\
        --input sweep
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import queue
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from audio.preprocessing.preprocess import (
    extract_mel_spectrogram,
    extract_wav2vec2_features,
    load_preprocess_config,
    load_wav2vec2_model,
)
from audio.training.dataset import resolve_samples
from audio.training.evaluate import load_checkpoint_model
from audio.training.metrics import compute_metrics
from audio.training.utils import find_project_root, load_training_config
from audio.calibration.temperature_scaling import CalibratedModel

logger = logging.getLogger(__name__)

# Sweep parameters (configurable via CLI flags)
SWEEP_WINDOWS = [1.0, 2.0, 3.0, 4.0]      # seconds
SWEEP_OVERLAPS = [0.0, 0.25, 0.50]         # fraction


# ---------------------------------------------------------------------------
# Core inference class
# ---------------------------------------------------------------------------

class AudioStreamingInference:
    """Sliding-window real-time inference with EMA score smoothing.

    Attributes:
        model: Trained PyTorch audio model.
        prep_config: PreprocessConfig from audio_preprocessing.yaml.
        feature_mode: ``"mel_spectrogram"`` or ``"wav2vec2"``.
        device: Torch inference device.
        window_sec: Chunk length in seconds (default 2 s).
        overlap_pct: Fraction overlap between consecutive windows (default 0.5).
        alpha: EMA decay for score smoothing (default 0.3).
        w2v_model: Optional loaded Wav2Vec2Model (only for wav2vec2 mode).
        w2v_processor: Optional loaded Wav2Vec2Processor.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        prep_config: Any,
        feature_mode: str,
        device: torch.device,
        *,
        window_sec: float = 2.0,
        overlap_pct: float = 0.5,
        alpha: float = 0.3,
        w2v_model: Any = None,
        w2v_processor: Any = None,
    ) -> None:
        if not (0.0 <= overlap_pct < 1.0):
            raise ValueError(f"overlap_pct must be in [0, 1). Got {overlap_pct}")
        if alpha <= 0.0 or alpha > 1.0:
            raise ValueError(f"alpha must be in (0, 1]. Got {alpha}")

        self.model = model
        self.prep_config = prep_config
        self.feature_mode = feature_mode
        self.device = device
        self.window_sec = window_sec
        self.overlap_pct = overlap_pct
        self.alpha = alpha
        self.w2v_model = w2v_model
        self.w2v_processor = w2v_processor

        self.sample_rate: int = prep_config.sample_rate
        self.chunk_samples: int = int(self.sample_rate * window_sec)
        # step = non-overlapping portion
        self.step_samples: int = max(1, int(self.chunk_samples * (1.0 - overlap_pct)))

        # Internal state reset per stream
        self._buffer: np.ndarray = np.zeros(0, dtype=np.float32)
        self._last_score: float | None = None
        self._is_running: bool = False

    # -----------------------------------------------------------------------
    # Chunk-level processing
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def process_chunk(self, waveform: np.ndarray) -> tuple[float, float]:
        """Extract features from a raw waveform chunk and run inference.

        Args:
            waveform: Float32 mono waveform at ``self.sample_rate``, length
                      exactly ``self.chunk_samples`` samples.

        Returns:
            Tuple of ``(raw_probability_fake, latency_ms)``.
        """
        t0 = time.perf_counter()
        waveform = waveform.astype(np.float32)

        if self.feature_mode == "mel_spectrogram":
            features = extract_mel_spectrogram(waveform, self.prep_config)
            # [n_mels, T] → [1, 1, n_mels, T]
            tensor = torch.from_numpy(features).unsqueeze(0).unsqueeze(0).to(self.device)
        else:
            if self.w2v_model is None or self.w2v_processor is None:
                raise RuntimeError("wav2vec2 model/processor not loaded.")
            features = extract_wav2vec2_features(
                waveform, self.prep_config, self.w2v_model, self.w2v_processor
            )
            # [T, D] → [1, T, D]
            tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        raw_prob = float(torch.sigmoid(logits).cpu().item())
        latency_ms = (time.perf_counter() - t0) * 1_000.0
        return raw_prob, latency_ms

    # -----------------------------------------------------------------------
    # Buffer / sliding window
    # -----------------------------------------------------------------------

    def push_audio(
        self,
        new_samples: np.ndarray,
        callback: Callable[[float, float], None] | None = None,
    ) -> None:
        """Append samples to the internal buffer and fire inference for every
        complete chunk available.

        Args:
            new_samples: 1-D float32 array of audio samples.
            callback: Optional ``(smoothed_score, latency_ms) → None`` called
                      for every emitted chunk.  Also prints to stdout.
        """
        self._buffer = np.concatenate([self._buffer, new_samples.astype(np.float32)])

        while len(self._buffer) >= self.chunk_samples:
            chunk = self._buffer[: self.chunk_samples]
            raw_score, latency = self.process_chunk(chunk)

            # EMA smoothing
            if self._last_score is None:
                smoothed = raw_score
            else:
                smoothed = self.alpha * raw_score + (1.0 - self.alpha) * self._last_score
            self._last_score = smoothed

            if callback is not None:
                callback(smoothed, latency)

            # Advance sliding window
            self._buffer = self._buffer[self.step_samples :]

    def reset(self) -> None:
        """Clear buffer and EMA state (call between files)."""
        self._buffer = np.zeros(0, dtype=np.float32)
        self._last_score = None

    # -----------------------------------------------------------------------
    # High-level streaming modes
    # -----------------------------------------------------------------------

    def stream_file(
        self,
        filepath: Path,
        callback: Callable[[float, float], None] | None = None,
    ) -> list[float]:
        """Simulate real-time streaming from a .wav / .flac file.

        Audio is resampled to ``self.sample_rate``, then pushed in 100 ms
        blocks to replicate a live stream.  Internal state is reset after.

        Args:
            filepath: Path to a WAV or FLAC file.
            callback: Optional ``(smoothed_score, latency_ms) → None``.

        Returns:
            List of per-chunk smoothed confidence scores.
        """
        try:
            import librosa
        except ImportError as exc:
            raise ImportError("librosa is required for file streaming.") from exc

        waveform, _ = librosa.load(str(filepath), sr=self.sample_rate, mono=True)

        scores: list[float] = []

        def _wrapped(score: float, lat: float) -> None:
            scores.append(score)
            if callback is not None:
                callback(score, lat)

        block_size = int(self.sample_rate * 0.1)  # 100 ms blocks
        for i in range(0, len(waveform), block_size):
            self.push_audio(waveform[i : i + block_size], _wrapped)

        self.reset()
        return scores

    def stream_mic(
        self,
        *,
        window_sec: float | None = None,
        overlap_pct: float | None = None,
    ) -> None:
        """Block and stream audio from the default microphone.

        Requires the ``sounddevice`` package.  Press Ctrl-C to stop.
        """
        try:
            import sounddevice as sd  # lazy import: only needed for mic mode
        except ImportError as exc:
            raise ImportError(
                "sounddevice is required for microphone streaming. "
                "Install it with: pip install sounddevice"
            ) from exc

        q: queue.Queue[np.ndarray] = queue.Queue()

        def _sd_callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
            if status:
                print(status, file=sys.stderr)
            q.put(indata.copy().ravel())

        def _score_cb(score: float, latency: float) -> None:
            bar_len = int(score * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            print(f"\r[{bar}] {score:.3f}  ({latency:.0f} ms)", end="", flush=True)

        print(f"Microphone streaming — window={self.window_sec}s, overlap={self.overlap_pct:.0%}")
        print("Press Ctrl-C to stop.\n")
        self._is_running = True
        self.reset()

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                callback=_sd_callback,
            ):
                while self._is_running:
                    samples = q.get()
                    self.push_audio(samples, _score_cb)
        except KeyboardInterrupt:
            pass
        finally:
            self._is_running = False
            self.reset()
            print("\nStopped.")


# ---------------------------------------------------------------------------
# Latency-vs-accuracy sweep
# ---------------------------------------------------------------------------

def latency_accuracy_sweep(
    model: torch.nn.Module,
    train_config: Any,
    prep_config: Any,
    device: torch.device,
    *,
    w2v_model: Any = None,
    w2v_processor: Any = None,
    windows: list[float] = SWEEP_WINDOWS,
    overlaps: list[float] = SWEEP_OVERLAPS,
) -> Path:
    """Run a grid sweep over (window_size × overlap) on ``test_seen``.

    For each configuration:
    - Load raw audio from original_path (via metadata.csv)
    - Simulate streaming, collect per-chunk latencies
    - Aggregate per-file scores (mean) → compute F1 / accuracy / roc_auc

    Saves results to:
    - ``reports/audio/latency_accuracy_curve.json``
    - ``reports/audio/latency_accuracy_curve.png``

    Returns:
        Path to the JSON output.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless backend
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for the sweep plot. pip install matplotlib") from exc

    logger.info("Starting latency vs. accuracy sweep...")
    test_csv = train_config.split_dir / "test_seen.csv"
    if not test_csv.is_file():
        raise FileNotFoundError(f"test_seen split not found: {test_csv}")

    samples = resolve_samples(
        test_csv,
        train_config.metadata_path,
        train_config.project_root,
        preprocessing_version=train_config.preprocessing_version,
        feature_mode=train_config.feature_mode,
    )
    if not samples:
        raise RuntimeError("No test_seen samples resolved; cannot run sweep.")

    # Build clip_id → original_path mapping from preprocessing metadata
    orig_path_map: dict[str, str] = {}
    if train_config.metadata_path.is_file():
        with train_config.metadata_path.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                cid = row.get("clip_id", "").strip()
                op = row.get("original_path", "").strip()
                if cid and op:
                    orig_path_map[cid] = op

    results: list[dict[str, Any]] = []

    for w in windows:
        for o in overlaps:
            logger.info("Sweep  window=%.1f s  overlap=%.0f%%", w, o * 100)
            streamer = AudioStreamingInference(
                model=model,
                prep_config=prep_config,
                feature_mode=train_config.feature_mode,
                device=device,
                window_sec=w,
                overlap_pct=o,
                w2v_model=w2v_model,
                w2v_processor=w2v_processor,
            )

            y_true: list[int] = []
            y_prob: list[float] = []
            chunk_latencies: list[float] = []

            for s in samples:
                orig_rel = orig_path_map.get(s.clip_id)
                if not orig_rel:
                    continue
                orig_abs = train_config.project_root / orig_rel
                if not orig_abs.is_file():
                    continue

                file_lats: list[float] = []

                def _lat_cb(score: float, lat: float, _fl: list = file_lats) -> None:
                    _fl.append(lat)

                scores = streamer.stream_file(orig_abs, callback=_lat_cb)
                if scores:
                    y_prob.append(float(np.mean(scores)))
                    y_true.append(s.label)
                    chunk_latencies.extend(file_lats)

            if not y_true:
                logger.warning("No files evaluated for window=%.1f overlap=%.0f%%", w, o * 100)
                continue

            metrics = compute_metrics(np.array(y_true, dtype=np.int64), np.array(y_prob))
            avg_lat = float(np.mean(chunk_latencies)) if chunk_latencies else 0.0

            results.append(
                {
                    "window_sec": w,
                    "overlap_pct": o,
                    "latency_ms": round(avg_lat, 3),
                    "accuracy": metrics.get("accuracy"),
                    "f1": metrics.get("f1"),
                    "roc_auc": metrics.get("roc_auc"),
                    "n_files": len(y_true),
                    "n_chunks": len(chunk_latencies),
                }
            )
            logger.info(
                "  acc=%.4f  f1=%.4f  roc_auc=%s  avg_latency=%.1f ms",
                metrics.get("accuracy") or 0.0,
                metrics.get("f1") or 0.0,
                metrics.get("roc_auc"),
                avg_lat,
            )

    # -- Save JSON -----------------------------------------------------------
    reports_dir = train_config.project_root / "reports" / "audio"
    reports_dir.mkdir(parents=True, exist_ok=True)

    json_path = reports_dir / "latency_accuracy_curve.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
        fh.write("\n")
    logger.info("Saved sweep JSON: %s", json_path)

    # -- Save PNG ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ["o", "s", "^"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for idx, o in enumerate(overlaps):
        subset = [r for r in results if r["overlap_pct"] == o]
        if not subset:
            continue
        lats = [r["latency_ms"] for r in subset]
        f1s = [r["f1"] or 0.0 for r in subset]
        label = f"Overlap {int(o * 100)}%"
        ax.plot(lats, f1s, marker=markers[idx % len(markers)],
                color=colors[idx % len(colors)], label=label, linewidth=2)
        for r in subset:
            ax.annotate(
                f"{r['window_sec']}s",
                xy=(r["latency_ms"], r["f1"] or 0.0),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    ax.set_xlabel("Avg latency per chunk (ms)")
    ax.set_ylabel("F1 Score")
    ax.set_title("Latency vs. Accuracy Sweep — Audio Baseline (test_seen)")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.6)

    png_path = reports_dir / "latency_accuracy_curve.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved sweep PNG:  %s", png_path)

    return json_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AEGIS audio streaming inference and latency-accuracy sweep.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained checkpoint (.pt).",
    )
    p.add_argument(
        "--input",
        type=str,
        required=True,
        metavar="FILE_OR_MIC",
        help="'mic' for microphone, 'sweep' for latency-accuracy sweep, "
             "or path to a .wav / .flac file.",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to audio_baseline.yaml (default: <project_root>/configs/audio_baseline.yaml).",
    )
    p.add_argument(
        "--prep-config",
        type=Path,
        default=None,
        help="Path to audio_preprocessing.yaml (default: <project_root>/configs/audio_preprocessing.yaml).",
    )
    p.add_argument(
        "--window",
        type=float,
        default=2.0,
        help="Sliding window size in seconds (default: 2.0).",
    )
    p.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Window overlap fraction 0–<1 (default: 0.5).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="EMA smoothing coefficient (default: 0.3).",
    )
    p.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="AEGIS project root (defaults to auto-discovery).",
    )
    p.add_argument(
        "--calibrated",
        action="store_true",
        default=False,
        help=(
            "Load baseline_calibrated.pt from the same directory as --checkpoint "
            "and apply temperature scaling. If the checkpoint itself carries a "
            "'temperature' key, applies it automatically."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)

    project_root = (args.project_root or find_project_root()).resolve()
    config_path = (args.config or project_root / "configs" / "audio_baseline.yaml").resolve()
    prep_config_path = (args.prep_config or project_root / "configs" / "audio_preprocessing.yaml").resolve()

    train_config = load_training_config(config_path, project_root)
    prep_config = load_preprocess_config(prep_config_path, project_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    checkpoint_path = args.checkpoint.resolve()
    if args.calibrated:
        calibrated_path = checkpoint_path.parent / "baseline_calibrated.pt"
        if calibrated_path.is_file():
            checkpoint_path = calibrated_path
            logger.info("--calibrated flag set → using %s", checkpoint_path)
        else:
            logger.warning(
                "--calibrated flag set but %s not found. "
                "Will attempt to apply temperature from the provided checkpoint.",
                calibrated_path,
            )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model, payload = load_checkpoint_model(checkpoint_path, device)

    checkpoint_T = payload.get("temperature", None)
    if checkpoint_T is not None and checkpoint_T != 1.0:
        logger.info(
            "Checkpoint carries temperature T=%.6f → wrapping model with CalibratedModel.",
            checkpoint_T,
        )
        model = CalibratedModel(model, temperature=checkpoint_T)
    elif args.calibrated:
        logger.warning(
            "--calibrated flag is set but checkpoint has no temperature key. "
            "Using T=1.0 (no calibration)."
        )

    model.eval()
    logger.info("Loaded checkpoint: %s", checkpoint_path)

    # Load wav2vec2 backbone lazily
    w2v_model, w2v_processor = None, None
    if train_config.feature_mode == "wav2vec2":
        logger.info("Loading wav2vec2 encoder for feature extraction...")
        w2v_model, w2v_processor = load_wav2vec2_model(prep_config)

    # -----------------------------------------------------------------------
    # Dispatch
    # -----------------------------------------------------------------------
    mode = args.input.strip().lower()

    if mode == "sweep":
        latency_accuracy_sweep(
            model=model,
            train_config=train_config,
            prep_config=prep_config,
            device=device,
            w2v_model=w2v_model,
            w2v_processor=w2v_processor,
        )

    elif mode == "mic":
        streamer = AudioStreamingInference(
            model=model,
            prep_config=prep_config,
            feature_mode=train_config.feature_mode,
            device=device,
            window_sec=args.window,
            overlap_pct=args.overlap,
            alpha=args.alpha,
            w2v_model=w2v_model,
            w2v_processor=w2v_processor,
        )
        streamer.stream_mic()

    else:
        filepath = Path(args.input).resolve()
        if not filepath.is_file():
            raise FileNotFoundError(f"Input file not found: {filepath}")

        streamer = AudioStreamingInference(
            model=model,
            prep_config=prep_config,
            feature_mode=train_config.feature_mode,
            device=device,
            window_sec=args.window,
            overlap_pct=args.overlap,
            alpha=args.alpha,
            w2v_model=w2v_model,
            w2v_processor=w2v_processor,
        )

        def stdout_cb(score: float, lat: float) -> None:
            bar_len = int(score * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            print(f"[{bar}] p(fake)={score:.4f}  ({lat:.0f} ms)")

        logger.info("Streaming file: %s", filepath)
        final_scores = streamer.stream_file(filepath, callback=stdout_cb)
        if final_scores:
            print(
                f"\n--- Summary ---\n"
                f"  Chunks processed : {len(final_scores)}\n"
                f"  Mean p(fake)     : {np.mean(final_scores):.4f}\n"
                f"  Max  p(fake)     : {np.max(final_scores):.4f}\n"
                f"  Verdict          : {'FAKE' if np.mean(final_scores) >= 0.5 else 'REAL'}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
