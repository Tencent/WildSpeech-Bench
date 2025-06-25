import json
import argparse
import statistics
import numpy as np
from scipy import stats
from langdetect import detect

# 判断两个字符串是否都是英文
def is_english(response):
    response_lang = detect(response)
    return response_lang == 'en'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_file', required=True)
    return parser.parse_args()

def calculate_stats(scores):
    if not scores:
        return {
            'mean': 0,
            'std_dev': 0,
            'conf_interval': (0, 0)
        }
    
    mean = np.mean(scores)
    std_dev = np.std(scores, ddof=1)  # ddof=1 for sample standard deviation
    
    # 95% confidence interval
    n = len(scores)
    confidence = 0.95
    degrees_freedom = n - 1
    t_value = stats.t.ppf((1 + confidence) / 2, degrees_freedom)
    margin_of_error = t_value * (std_dev / np.sqrt(n))

    return {
        'mean': mean,
        'std_dev': std_dev,
        'conf_interval': (mean - margin_of_error, mean + margin_of_error)
    }

if __name__ == "__main__":
    args = parse_args()

    with open(args.src_file, 'r') as f:
        mos_scores = []
        gpt_scores = []
        gpt_score_natural_noise = []
        gpt_score_human_noise = []
        gpt_score_clean = []
        gpt_scores_by_task_type = {}  # Dictionary to store scores by taskType
        gpt_scores_by_task_type_pro = {}  # Dictionary to store scores by taskType
        mos_scores_by_task_type = {}  # Dictionary to store MOS scores by taskType
        
        for line in f:
            item = json.loads(line)
            response = item['response'][0]
            try:
                is_english_res =  is_english(response)
                # is_english_res =  is_english(response)
            except Exception as e:
                print(f'exception: {e}')
                is_english_res = True
            if "score" in item and isinstance(item['score'], list):
                gpt_scores.append(np.mean(item['score']) if is_english_res else 1)
                task_type = item["taskType"]
                if task_type not in gpt_scores_by_task_type:
                    gpt_scores_by_task_type[task_type] = []
                gpt_scores_by_task_type[task_type].append(np.mean(item['score']) if is_english_res else 1)

                if task_type == 'prosodical':
                    task_type_pro = item["additional_info"]
                    if "stuttering" in task_type_pro:
                        task_type_pro = "stuttering"
                    if task_type_pro not in gpt_scores_by_task_type_pro:
                        gpt_scores_by_task_type_pro[task_type_pro] = []
                    gpt_scores_by_task_type_pro[task_type_pro].append(np.mean(item['score']) if is_english_res else 1)
                    
            if "mos" in item and isinstance(item['mos'], float):
                mos_scores.append(item['mos'])
                task_type = item["taskType"]
                if task_type not in mos_scores_by_task_type:
                    mos_scores_by_task_type[task_type] = []
                mos_scores_by_task_type[task_type].append(item['mos'])
            if "modification_type" in item and "score" in item and isinstance(item['score'], list):
                if item['modification_type'] == 'natural_noise':
                    gpt_score_natural_noise.append(np.mean(item['score']) if is_english_res else 1)
                elif item['modification_type'] == 'human_noise':
                    gpt_score_human_noise.append(np.mean(item['score']) if is_english_res else 1)
                else:
                    gpt_score_clean.append(np.mean(item['score']) if is_english_res else 1)

    # print(f'gpt_scores: {sum(gpt_scores) / len(gpt_scores)}')
    # print(f'mos_scores: {sum(mos_scores) / len(mos_scores)}')

    # print(f'gpt_score_natural_noise: {sum(gpt_score_natural_noise) / len(gpt_score_natural_noise)}')
    # print(f'gpt_score_human_noise: {sum(gpt_score_human_noise) / len(gpt_score_human_noise)}')
    # print(f'gpt_score_clean: {sum(gpt_score_clean) / len(gpt_score_clean)}')

    print(f'gpt_scores stats:')
    print(f'length: {len(gpt_scores)}')
    results1 = calculate_stats(gpt_scores)
    print(f'  Mean: {results1["mean"]:.4f}')
    print(f'  Std Dev: {results1["std_dev"]:.4f}')
    print(f'  95% CI: ({results1["conf_interval"][0]:.4f}, {results1["conf_interval"][1]:.4f})')

    print(f'mos_scores stats:')
    results2 = calculate_stats(mos_scores)
    print(f'  Mean: {results2["mean"]:.4f}')
    print(f'  Std Dev: {results2["std_dev"]:.4f}')
    print(f'  95% CI: ({results2["conf_interval"][0]:.4f}, {results2["conf_interval"][1]:.4f})')

    print(f'gpt_score_natural_noise stats:')
    results3 = calculate_stats(gpt_score_natural_noise)
    print(f'  Mean: {results3["mean"]:.4f}')
    print(f'  Std Dev: {results3["std_dev"]:.4f}')
    print(f'  95% CI: ({results3["conf_interval"][0]:.4f}, {results3["conf_interval"][1]:.4f})')

    print(f'gpt_score_human_noise stats:')
    results4 = calculate_stats(gpt_score_human_noise)
    print(f'  Mean: {results4["mean"]:.4f}')
    print(f'  Std Dev: {results4["std_dev"]:.4f}')
    print(f'  95% CI: ({results4["conf_interval"][0]:.4f}, {results4["conf_interval"][1]:.4f})')

    print(f'gpt_score_clean stats:')
    results5 = calculate_stats(gpt_score_clean)
    print(f'  Mean: {results5["mean"]:.4f}')
    print(f'  Std Dev: {results5["std_dev"]:.4f}')
    print(f'  95% CI: ({results5["conf_interval"][0]:.4f}, {results5["conf_interval"][1]:.4f})')

    # After printing other stats, add stats by taskType
    print("\nScores by Task Type:")
    for task_type, scores in gpt_scores_by_task_type.items():
        print(f'\n{task_type} stats:')
        print(f'length: {len(scores)}')
        results = calculate_stats(scores)
        print(f'  Mean: {results["mean"]:.4f}')
        print(f'  Std Dev: {results["std_dev"]:.4f}')
        print(f'  95% CI: ({results["conf_interval"][0]:.4f}, {results["conf_interval"][1]:.4f})')
        
    # Add MOS scores by task type
    print("\nMOS Scores by Task Type:")
    for task_type, scores in mos_scores_by_task_type.items():
        print(f'\n{task_type} MOS stats:')
        print(f'length: {len(scores)}')
        results = calculate_stats(scores)
        print(f'  Mean: {results["mean"]:.4f}')
        print(f'  Std Dev: {results["std_dev"]:.4f}')
        print(f'  95% CI: ({results["conf_interval"][0]:.4f}, {results["conf_interval"][1]:.4f})')