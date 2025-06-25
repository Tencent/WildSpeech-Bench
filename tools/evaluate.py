from argparse import ArgumentParser
import json
from src.evaluator.open import OpenEvaluator


def main():
    parser = ArgumentParser()
    parser.add_argument('--src_file', type=str, required=True)
    parser.add_argument('--results_csv', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')
    args = parser.parse_args()

    data = []
    with open(args.src_file, 'r') as f:
        for line in f:
            json_obj = json.loads(line.strip())  # Convert JSON string to dictionary
            data.append(json_obj)
    evaluator = OpenEvaluator()
    results = evaluator.evaluate(data)
    print(results)
    with open(args.results_csv, 'a') as f:
        for k, v in results.items():
            f.write(f'{args.dataset_name}-{args.split},{k},{v}\n')


if __name__ == "__main__":
    main()
