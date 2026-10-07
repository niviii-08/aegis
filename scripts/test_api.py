"""Test the AEGIS API endpoints."""
import argparse
import time
import requests
from pathlib import Path

def wait_for_api(url, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{url}/health")
            if r.status_code == 200:
                print("API is healthy!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    print("API failed to start or become healthy.")
    return False

def test_models(url):
    r = requests.get(f"{url}/models")
    print("\n--- Models ---")
    print(r.json())

def test_multimodal(url, sample_image=None, sample_audio=None):
    print("\n--- Multimodal Prediction ---")
    files = {}
    if sample_image and Path(sample_image).exists():
        files['image'] = open(sample_image, 'rb')
    if sample_audio and Path(sample_audio).exists():
        files['audio'] = open(sample_audio, 'rb')
        
    if not files:
        print("No sample files found to test.")
        return
        
    r = requests.post(f"{url}/predict/multimodal", files=files)
    print(f"Status: {r.status_code}")
    print(r.json())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--image", help="Path to test image")
    parser.add_argument("--audio", help="Path to test audio")
    args = parser.parse_args()
    
    if wait_for_api(args.url):
        test_models(args.url)
        test_multimodal(args.url, args.image, args.audio)
