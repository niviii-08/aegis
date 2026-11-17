"""AEGIS Video Deepfake Forensic Inspector & Temporal Attribution Analyzer.

Features:
1. Calibrated Neural Inference: Real/Fake classification using optimal decision threshold.
2. Temporal Frame Attribution: Identifies the exact video frame with peak forgery energy.
3. Spatial Grad-CAM Localization: Generates side-by-side activation maps highlighting facial forgeries.
4. Test-Time Augmentation (TTA): Ensemble prediction over spatial and temporal flipped sequences.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from api.services.video_service import VideoInferenceService
from src.video.analysis.gradcam import process_video_gradcam

DEFAULT_MODEL = PROJECT_ROOT / "models" / "video" / "celebdf_best.pt"


def render_ascii_heatmap(scores: List[float], peak_idx: int) -> str:
    """Render a visual ASCII bar chart of per-frame anomaly energy."""
    bars = []
    max_s = max(scores) if scores else 1.0
    for i, s in enumerate(scores):
        bar_len = int((s / max(max_s, 1e-4)) * 12)
        bar_char = "#" if i == peak_idx else "="
        marker = "<-- PEAK ANOMALY" if i == peak_idx else ""
        bars.append(f"  Frame {i+1:02d} | [{bar_char * bar_len:<12}] {s*100:5.1f}% {marker}")
    return "\n".join(bars)


def analyze_video(
    svc: VideoInferenceService,
    video_path: Path,
    model_path: Path,
    gradcam: bool = False,
    tta: bool = False,
) -> dict[str, Any]:
    """Run complete forensic analysis on a single video."""
    res = svc.predict(video_path, tta=tta)
    meta = res.get("explanation_metadata") or res.get("metadata", {})
    scores = meta.get("frame_scores", [])
    peak_idx = meta.get("peak_frame_index", 0)

    print("\n" + "=" * 80)
    print(f"  FORENSIC VIDEO INSPECTION: {video_path.name}")
    print("=" * 80)
    print(f"  Verdict          : {res['prediction'].upper()}")
    print(f"  Forgery Prob     : {res['probability']*100:.2f}% (P_fake: {res['probability']:.4f})")
    print(f"  Calibrated Thresh: {meta.get('threshold', 0.50):.3f}")
    print(f"  TTA Ensemble     : {'Enabled (3x Views)' if tta else 'Standard (1x View)'}")
    print(f"  Inference Latency: {res.get('inference_latency_ms', 0):.1f} ms")

    if scores:
        print(f"\n  TEMPORAL FRAME ATTRIBUTION (16 Uniform Samples):")
        print(f"  Peak Anomaly Detected at: Frame #{peak_idx + 1}")
        print(render_ascii_heatmap(scores, peak_idx))

    if gradcam:
        try:
            out_dir = PROJECT_ROOT / "reports" / "forensics"
            out_dir.mkdir(parents=True, exist_ok=True)
            gc_path = process_video_gradcam(video_path, model_path, out_dir, torch.device("cpu"), max_frames=8)
            print(f"\n  SPATIAL GRAD-CAM HEATMAP:")
            print(f"  Visual Map Saved -> {gc_path}")
        except Exception as e:
            print(f"\n  [Notice] Grad-CAM generation: {e}")

    print("=" * 80 + "\n")
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="AEGIS Video Deepfake Forensic Inspector")
    parser.add_argument("videos", nargs="*", help="Video file paths to inspect")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Model checkpoint path")
    parser.add_argument("--gradcam", action="store_true", help="Generate spatial Grad-CAM saliency heatmaps")
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation (Spatial & Temporal Flipped Ensemble)")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"Error: Model checkpoint not found at {model_path}")
        sys.exit(1)

    print(f"Loading AEGIS Video Forensic Engine from {model_path.name}...")
    svc = VideoInferenceService(checkpoint_path=model_path, device="cpu")

    if args.videos:
        video_paths = [Path(v) for v in args.videos]
    else:
        # Default: sample authentic real and sample synthesis deepfake
        data_dir = PROJECT_ROOT / "data" / "raw" / "video" / "celeb_df_v2"
        reals = sorted(list((data_dir / "Celeb-real").glob("*.mp4")))[:1]
        fakes = sorted(list((data_dir / "Celeb-synthesis").glob("*.mp4")))[:1]
        video_paths = reals + fakes
        print(f"No video paths passed. Running forensic inspection on {len(video_paths)} benchmark samples...\n")

    for v in video_paths:
        if v.is_file():
            analyze_video(svc, v, model_path, gradcam=args.gradcam, tta=args.tta)
        else:
            print(f"Warning: File not found {v}")


if __name__ == "__main__":
    main()
