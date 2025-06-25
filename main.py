import torchaudio
import json
import os
from datasets import load_dataset
from argparse import ArgumentParser
from tqdm import tqdm
from tools.cal_mos import cal_mos_single
from tools.transcribe_multi import transcribe_audio
from src.models import model_cls_mapping

REQUIRED_FIELDS = [
    'user_query',
    'taskType',
    'checklist',
    'conversation_hash',
    'modification_type',
    'additional_info'
]

def extract_required_fields(item):
    """Extract required fields from the item dictionary."""
    return {k: v for k, v in item.items() if k in REQUIRED_FIELDS}

def main():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, choices=list(model_cls_mapping.keys()))
    parser.add_argument('--result_dir', type=str)
    args = parser.parse_args()
    target_subdir = "wavs"  # wavs
    output_jsonl_name = f'{args.model}_results.jsonl'
    os.makedirs(f'{args.result_dir}/{target_subdir}', exist_ok=True)
    data = load_dataset('tencent/WildSpeech-Bench', split='train')
    model = model_cls_mapping[args.model]()
    with open(f'{args.result_dir}/{output_jsonl_name}', 'w') as f:
        for idx, item in enumerate(tqdm(data, total=len(data))):
            tmp = extract_required_fields(item)
            audio, sr = model.generate_audio(item['audio'])
            torchaudio.save(f'{args.result_dir}/{target_subdir}/{idx}.wav', audio, sr)
            transcribed_texts = transcribe_audio(f'{args.result_dir}/{target_subdir}/{idx}.wav')
            tmp['response'] = transcribed_texts
            tmp['mos'] = cal_mos_single(f'{args.result_dir}/{target_subdir}/{idx}.wav')
            json_line = json.dumps(tmp)
            f.write(json_line + '\n')
            f.flush()
if __name__ == '__main__':
    main()
