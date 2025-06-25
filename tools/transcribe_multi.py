import whisper
import re
import json
import os
import torchaudio
from tqdm import tqdm
import sys
print(whisper.__file__)

asr_model = whisper.load_model("large-v3").to("cuda")

def transcribe_audio(audio_path, max_transcribe_times=10, target_times=3):
    transcribed_texts = []
    all_transcriptions = []  # Store all transcriptions in case we need them
    for _ in range(max_transcribe_times):
        transcribed_text = asr_model.transcribe(audio_path, task="transcribe")['text']
        all_transcriptions.append(transcribed_text)
        if not has_consecutive_repeated_sentence(transcribed_text):
            transcribed_texts.append(transcribed_text)
        else:
            print(f'bad transcribed_text: {transcribed_text}')
        if len(transcribed_texts) >= target_times:
            break
    if not transcribed_texts and all_transcriptions:
        return all_transcriptions[:target_times]
    return transcribed_texts

def has_consecutive_repeated_sentence(text, min_repeat=3, min_len=4): # account for whisper hallucination
    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip().lower() for s in sentences if len(s.strip()) >= min_len]
    count = 1
    for i in range(1, len(sentences)):
        if sentences[i] == sentences[i-1]:
            count += 1
            if count >= min_repeat:
                return True
        else:
            count = 1
    return False
