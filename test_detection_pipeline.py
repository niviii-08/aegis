"""Test and verify detection across Image, Audio, and Video modalities end-to-end."""

import sys
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from api.services.image_service import ImageInferenceService
from api.services.audio_service import AudioInferenceService
from api.services.video_service import VideoInferenceService

def test_image_detection():
    print("\n--- 1. Testing Image Detection ---")
    checkpoint = project_root / "models/image/baseline_best.pt"
    service = ImageInferenceService(checkpoint, device="cpu", threshold=0.5)
    
    sample_img = project_root / "data/processed/image/preprocessing/crops/real_vs_fake__test__00001.jpg"
    print(f"Sample image: {sample_img}")
    preprocessed = service.preprocess(sample_img)
    print(f"Preprocessed image tensor shape: {preprocessed.shape}")
    
    result = service.predict(preprocessed)
    print(f"Image detection result: prediction={result['prediction']}, probability={result['probability']:.4f}, latency={result['inference_latency_ms']:.1f}ms")
    assert result['prediction'] in ["real", "fake"]
    assert 0.0 <= result['probability'] <= 1.0
    print("Image Detection: PASSED")
    return result

def test_audio_detection():
    print("\n--- 2. Testing Audio Detection ---")
    checkpoint = project_root / "models/audio/baseline_best.pt"
    service = AudioInferenceService(checkpoint, device="cpu", threshold=0.5)
    
    sample_audio = project_root / "data/processed/audio/mel_spectrogram/A01/LA_D_1000265.npy"
    print(f"Sample audio: {sample_audio}")
    preprocessed = service.preprocess(sample_audio)
    print(f"Preprocessed audio tensor shape: {preprocessed.shape}")
    
    result = service.predict(preprocessed)
    print(f"Audio detection result: prediction={result['prediction']}, probability={result['probability']:.4f}, latency={result['inference_latency_ms']:.1f}ms")
    assert result['prediction'] in ["real", "fake"]
    assert 0.0 <= result['probability'] <= 1.0
    print("Audio Detection: PASSED")
    return result

def test_video_detection():
    print("\n--- 3. Testing Video Detection ---")
    checkpoint = project_root / "models/video/baseline_best.pt"
    service = VideoInferenceService(checkpoint, device="cpu", threshold=0.5)
    
    sample_video = project_root / "FaceForensics++_C23/DeepFakeDetection/01_02__meeting_serious__YVGY8LOK.mp4"
    print(f"Sample video: {sample_video}")
    preprocessed = service.preprocess(sample_video)
    print(f"Preprocessed video tensor shape: {preprocessed.shape}")
    
    result = service.predict(preprocessed)
    print(f"Video detection result: prediction={result['prediction']}, probability={result['probability']:.4f}, latency={result['inference_latency_ms']:.1f}ms")
    assert result['prediction'] in ["real", "fake"]
    assert 0.0 <= result['probability'] <= 1.0
    print("Video Detection: PASSED")
    return result

if __name__ == "__main__":
    img_res = test_image_detection()
    aud_res = test_audio_detection()
    vid_res = test_video_detection()
    print("\n" + "="*50)
    print("ALL THREE MODALITIES (IMAGE, AUDIO, VIDEO) ARE WORKING CORRECTLY!")
    print("="*50)
