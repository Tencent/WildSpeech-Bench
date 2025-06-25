import torch
import librosa
import argparse
import os
import numpy as np
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", type=str, required=True)
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    return parser.parse_args()

def cal_mos_single(audio_path):
    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    wave, sr = librosa.load(audio_path, sr=None, mono=True)
    mos_score = float(predictor(torch.from_numpy(wave).unsqueeze(0), sr).item())
    return mos_score

