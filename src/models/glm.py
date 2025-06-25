
import torch
import numpy as np
import whisper

from .base import VoiceAssistant
from .src_glm.speech_tokenizer.utils import extract_speech_token
from .src_glm.speech_tokenizer.modeling_whisper import WhisperVQEncoder
from .src_glm.flow_inference import AudioDecoder

from transformers import AutoModel, AutoTokenizer
from transformers import WhisperFeatureExtractor, AutoTokenizer

class GLMAssistant(VoiceAssistant):
    def __init__(self):
        model_path = 'THUDM/glm-4-voice-9b'
        self.glm_model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            quantization_config=None,
            torch_dtype=torch.bfloat16,
        ).eval().to("cuda")
        self.glm_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained("THUDM/glm-4-voice-tokenizer")
        self.whisper_model = WhisperVQEncoder.from_pretrained("THUDM/glm-4-voice-tokenizer").eval().to("cuda")
        self.asr_model = whisper.load_model("large-v3").to("cuda")
        # you should download the decoder from https://huggingface.co/THUDM/glm-4-voice-decoder
        self.audio_decoder = AudioDecoder(
            config_path="./src/models/src_glm/glm-4-voice-decoder/config.yaml",
            flow_ckpt_path="./src/models/src_glm/glm-4-voice-decoder/flow.pt",
            hift_ckpt_path="./src/models/src_glm/glm-4-voice-decoder/hift.pt",
            device="cuda")
        self.audio_0_id = self.glm_tokenizer.convert_tokens_to_ids('<|audio_0|>')

    def generate_audio(
        self,
        audio,
        max_new_tokens=4096,
    ):
        audio_array, sr = audio['array'], audio['sampling_rate']
        audio_tokens = extract_speech_token(
            self.whisper_model, self.feature_extractor, [tuple([torch.from_numpy(audio_array).unsqueeze(0), sr])]
        )[0]
        assert len(audio_tokens) != 0
        audio_tokens = "".join([f"<|audio_{x}|>" for x in audio_tokens])
        audio_tokens = "<|begin_of_audio|>" + audio_tokens + "<|end_of_audio|>"
        user_input = audio_tokens
        system_prompt = "User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. "

        inputs = f"<|system|>\n{system_prompt}"
        inputs += f"<|user|>\n{user_input}<|assistant|>streaming_transcription\n"
        inputs = self.glm_tokenizer([inputs], return_tensors="pt")
        inputs = inputs.to('cuda')
        rtn = self.glm_model.generate(**inputs, max_new_tokens=max_new_tokens)[:, inputs.input_ids.size(1):]
        audio_array, sr = self._generate_output_audio(rtn[0], self.audio_decoder)
        return audio_array, sr

    
    def _generate_output_audio(self,all_tokens, audio_decoder):
        output_audio_tokens = [t - self.audio_0_id for t in all_tokens if t >= self.audio_0_id]
        if len(output_audio_tokens) == 0:
            return None, 22050
        output_audio_data, sr = self._tokens_to_audio(output_audio_tokens, audio_decoder)
        return output_audio_data, sr

    def _tokens_to_audio(self,token_ids, audio_decoder):
        with torch.cuda.amp.autocast():
            tokens = torch.tensor(token_ids, dtype=torch.int32, device='cuda').unsqueeze(0)
            tts_speech = audio_decoder.offline_inference(tokens)
        return tts_speech, 22050