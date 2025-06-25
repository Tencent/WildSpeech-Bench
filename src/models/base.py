import torch
import time
import re
import whisper

class VoiceAssistant:
    def __init__(self):
        self.asr_model = whisper.load_model("large-v3").to("cuda")

    @torch.no_grad()
    def generate_audio(
        self,
        audio,
        max_new_tokens=2048,
    ):
        raise NotImplementedError