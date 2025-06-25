import re
from .base import Evaluator
import numpy as np


def extract_rating(llm_output):
    pattern = r"\[\[(\d+)\]\]"
    match = re.search(pattern, llm_output)
    if match:
        return int(match.group(1))
    else:
        raise NotImplementedError


class OpenEvaluator(Evaluator):
    def evaluate(self, data):
        scores = []
        for item in data:
            for score in item['score']:
                try:
                    score = float(score)
                except Exception as e:
                    score = extract_rating(score)
                scores.append(score)
        return {'gpt': np.mean(scores)}

