@echo off
venv\Scripts\python.exe patch_manifest.py
set PYTHONPATH=src
venv\Scripts\python.exe -m audio.preprocessing.preprocess --config configs/audio_preprocessing.yaml
venv\Scripts\python.exe -m audio.splits.generator_split
