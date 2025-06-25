import soundfile as sf
import torch
import whisper
from .base import VoiceAssistant
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import tempfile


class QwenOmniAssistant(VoiceAssistant):
    def __init__(self):
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-7B", 
                                                                         device_map="cuda").eval()
        self.processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
        self.asr_model = whisper.load_model("large-v3").to("cuda")

    def generate_audio(
        self,
        audio,
    ):
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".wav") as temp_file:
            temp_filename = temp_file.name
            sf.write(temp_file.name, audio['array'], audio['sampling_rate'], format='wav')

        messages = [
            {"role": "system", "content": [
                {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."} 
            ], },
            {"role": "user", "content": [
                {"type": "audio", "audio": temp_filename},
            ]
             },
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        with torch.no_grad():
            output = self.model.generate(**inputs, use_audio_in_video=True, return_audio=True)
            audio_tensor = output[1]
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        return audio_tensor, 24000,