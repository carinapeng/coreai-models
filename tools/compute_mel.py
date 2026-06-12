"""Compute Whisper mel spectrogram from an audio file and save as raw float32."""
import sys
import numpy as np
from pathlib import Path
from scipy.signal import resample_poly
from math import gcd
import soundfile as sf
from transformers import AutoProcessor

MODEL_NAME = "openai/whisper-large-v3-turbo"
TARGET_SR = 16000

def compute_mel(audio_path: str, out_path: str) -> None:
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        g = gcd(sr, TARGET_SR)
        audio = resample_poly(audio, TARGET_SR // g, sr // g).astype(np.float32)

    feat = processor.feature_extractor(audio, sampling_rate=TARGET_SR, return_tensors="np")
    mel = feat["input_features"][0]  # (128, 3000) float32

    mel.tofile(out_path)
    print(f"Saved mel {mel.shape} → {out_path}")

if __name__ == "__main__":
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/jfk.flac"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/jfk_mel.bin"
    compute_mel(audio_path, out_path)
