import transformers
import torch
import soundfile as sf
import whisper

from .base import VoiceAssistant
from transformers import AutoModel, AutoTokenizer

class MiniCPMAssistant(VoiceAssistant):
    def __init__(self):
        self.model = AutoModel.from_pretrained(
            'openbmb/MiniCPM-o-2_6',
            trust_remote_code=True,
            attn_implementation='sdpa', # sdpa or flash_attention_2
            torch_dtype=torch.bfloat16,
            init_vision=True,
            init_audio=True,
            init_tts=True
        )
        self.model.init_tts()
        self.model.tts.float()
        self.model = self.model.eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-o-2_6', trust_remote_code=True)
        self.sys_prompt = self.model.get_sys_prompt(mode='audio_assistant', language='en')
        self.asr_model = whisper.load_model("large-v3").to("cuda")

    def generate_audio(
        self,
        audio,
        max_new_tokens=4096,
    ): 
        user_question = {'role': 'user', 'content': [audio['array']]}
        msgs = [self.sys_prompt, user_question]
        
        res = self.model.chat(
            msgs=msgs,
            tokenizer=self.tokenizer,
            sampling=True,
            max_new_tokens=max_new_tokens,
            use_tts_template=True,
            generate_audio=True,
            temperature=0.3,
        )
        audio_array = res['audio_wav'].unsqueeze(0)
        sampling_rate = res['sampling_rate']
        return audio_array, sampling_rate