import requests
import json
import io
import os

API_URL = "http://localhost:8000/predict/image"

real_path = r"data\raw\image\real_vs_fake\real-vs-fake\valid\real\27982.jpg"
fake_path = r"data\raw\image\real_vs_fake\real-vs-fake\valid\fake\POGFJRQ8F4.jpg"

def test_image(path):
    print(f"Testing {path}...")
    try:
        with open(path, "rb") as f:
            files = {"file": open(path, "rb")}
            response = requests.post(API_URL, files=files)
            
        print(f"Status Code: {response.status_code}")
        try:
            print(json.dumps(response.json(), indent=2))
        except:
            print(response.text)
    except Exception as e:
        print(f"Error: {e}")

test_image(real_path)
print("-" * 50)
test_image(fake_path)
