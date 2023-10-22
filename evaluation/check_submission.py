import argparse
import json
import re
from collections import defaultdict
import math

TASKS = ["mt_zh_en", "mt_en_fr", "mt_en_de", "mt_en_ru", "dialogue", "summarization", "stance", "safety", "intent"]
TASK_SIZES = {
    "dialogue": 3548,
    "summarization": 1805,
    "intent": 2440,
    "safety": 2826,
    "stance": 1722,
    "mt_zh_en": 1200,
    "mt_en_fr": 1000,
    "mt_en_de": 1000,
    "mt_en_ru": 1000
}

class BCOLORS:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def read_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def check_submission_file(submission_file):
    print("Checking submission file: " + BCOLORS.BOLD + submission_file + BCOLORS.ENDC, end=" ")

    submission = read_json(submission_file)
    task_size_counter = defaultdict(int)
    
    unique_example_ids = set()

    try:
        for i, sample in enumerate(submission):
            percent_done = math.floor(i * 100 / len(submission))
            if percent_done % 100 == 0:
                print(f".", end=" ", flush=True)

            if "task" not in sample:
                raise ValueError(f"`task` field not found in the following sample: \n{sample}")
            if "example_id" not in sample:
                raise ValueError(f"`example_id` field not found in the following sample: \n{sample}")
            if "prediction" not in sample:
                raise ValueError(f"`prediction` field not found in the following sample: \n{sample}")
            
            task = sample["task"]
            example_id = sample["example_id"]
            prediction = sample["prediction"]

            if not isinstance(task, str):
                raise ValueError(f"`task` field in the following sample is expected to be a string:\n{sample}")

            if not isinstance(example_id, str):
                raise ValueError(f"`example_id` field in following sample is expected to be a string:\n{sample}")
            
            if not isinstance(prediction, int):
                raise ValueError(f"`prediction` field in the following sample is expected to be an integer:\n{sample}")
            
            if task not in TASKS:
                raise ValueError(f"Task {task} not recognized as valid CRoW task in the following sample. Please use the task name from the provided benchmark data.\n{sample}")
            
            sha256_hash_regex = r"^[a-fA-F0-9]{64}$"
            
            if not re.match(sha256_hash_regex, example_id):
                raise ValueError(f"example_id {example_id} not recognized as valid CRoW example_id in the following sample. Please use the example_id from the provided benchmark data.\n{sample}")

            if prediction not in [0, 1]:
                raise ValueError(f"Prediction {prediction} not recognized as valid CRoW prediction in the following sample. Expected 0 or 1.\n{sample}")

            if example_id in unique_example_ids:
                raise ValueError(f"Found duplicated example_id {example_id} in the following sample. Please make sure that each example_id is unique and comes from the provided benchmark data.\n{sample}")

            unique_example_ids.add(example_id)
            task_size_counter[task] += 1
        
        for task in task_size_counter.keys():
            if task_size_counter[task] != TASK_SIZES[task]:
                raise ValueError(f"Task {task} has {task_size_counter[task]} samples in the submission file. Expected {TASK_SIZES[task]} samples.")
    except Exception as e:
        print(BCOLORS.FAIL + "FAILED" + BCOLORS.ENDC)
        raise e
    
    print(BCOLORS.OKGREEN + "OK" + BCOLORS.ENDC)

def main():
    parser = argparse.ArgumentParser(description="Check submission file(s) for CRoW leaderboard")
    parser.add_argument("submission_files", type=str, nargs="+", default=[], help="Path to submission file(s) in json")

    args = parser.parse_args()

    for submission_file in args.submission_files:
        check_submission_file(submission_file)
    
    print(BCOLORS.OKGREEN + "All submission files are valid!" + BCOLORS.ENDC)

if __name__ == '__main__':
    main()