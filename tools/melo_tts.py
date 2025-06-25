from melo.api import TTS
import re
import torch
import time
import torchaudio

tts_zh_model = None
tts_en_model = None

def initialize_tts_models(max_retries=5, initial_wait=10):
    """按需初始化TTS模型"""
    global tts_zh_model, tts_en_model
    
    # 如果模型已初始化，直接返回
    if tts_en_model is not None and tts_zh_model is not None:
        return
    
    for i in range(max_retries):
        try:
            tts_en_model = TTS(language='EN', device='auto')
            tts_zh_model = TTS(language='ZH', device='auto')
            return
        except Exception as e:
            print(f"初始化TTS模型失败，重试第{i+1}次")
            print(f"错误信息: {str(e)}")
            wait_time = initial_wait * (i + 1)
            print(f"等待{wait_time}秒后重试")
            time.sleep(wait_time)
    
    raise Exception("初始化TTS模型失败，已达到最大重试次数")

def detect_language(text):
    if re.search('[\u4e00-\u9fff]', text):
        return 'ZH' # 中英混合判断中文
    return 'EN'

def melo_tts(text):
    # 确保模型已初始化
    initialize_tts_models()
    
    language = detect_language(text)
    speed = 1.0
    model = tts_zh_model if language == 'ZH' else tts_en_model
    speaker_ids = model.hps.data.spk2id
    if language == 'ZH':
        audio_array = model.tts_to_file(text, speaker_ids[language], speed=speed, quiet=True)
    else: # EN
        audio_array = model.tts_to_file(text, speaker_ids['EN-US'], speed=speed, quiet=True)
    print(f'audio_array: {audio_array}')
    print(f'audio_array: {audio_array.shape}')
    return (audio_array, 44100)