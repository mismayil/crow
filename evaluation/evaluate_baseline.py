import sys
import pathlib
import argparse
import random 
import os
from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
sys.path.append("..")

from utils import read_json, write_json, generate_unique_id

def evaluate_majority(datapath, *args, **kwargs):
    references = []
    predictions = []

    num_positive_class = 0
    majority_label = 0
    
    data = read_json(datapath)

    for sample in tqdm(data, total=len(data), desc="Identifying majority label"):
        if sample["answer"].lower() == "yes":
            references.append(1)
            num_positive_class += 1
        else: 
            references.append(0)

    if num_positive_class > (len(data) / 2): 
        majority_label = 1
    elif num_positive_class == (len(data) / 2): 
        majority_label = random.randint(0, 1)
    else: 
        majority_label = 0

    for sample in tqdm(data, total=len(data), desc="Assigning predictions"):
        sample["response"] = "yes" if majority_label == 1 else "no"
        sample["correct"] = True if sample["response"] == sample["answer"].lower() else False
    
    predictions = [majority_label] * len(data)

    outputs = {
        "metadata": {
            "datapath": datapath,
            "model": "majority_baseline",
            "majority_label": "yes" if majority_label == 1 else "no"
        },
        "metrics": {
            "accuracy": accuracy_score(references, predictions),
            "precision": precision_score(references, predictions, average="macro"),
            "recall": recall_score(references, predictions, average="macro"),
            "f1": f1_score(references, predictions, average="macro")
        },
        "data": data
    }

    return outputs

def evaluate_random(datapath, seed=None, *args, **kwargs):
    references = []
    predictions = []

    if seed is None:
        seed = random.randint(0, 2 ** 32)

    random.seed(seed)

    data = read_json(datapath)

    for sample in tqdm(data, total=len(data), desc="Assigning predictions"):
        prediction = random.choice(["yes", "no"])
        references.append(1 if sample["answer"].lower() == "yes" else 0)
        predictions.append(1 if prediction == "yes" else 0)
        sample["response"] = prediction
        sample["correct"] = True if prediction == sample["answer"].lower() else False

    outputs = {
        "metadata": {
            "datapath": datapath,
            "model": "random_baseline",
            "seed": seed
        },
        "metrics": {
            "accuracy": accuracy_score(references, predictions),
            "precision": precision_score(references, predictions, average="macro"),
            "recall": recall_score(references, predictions, average="macro"),
            "f1": f1_score(references, predictions, average="macro")
        },
        "data": data
    }

    return outputs

MODELS_MAP = {
    "random": evaluate_random,
    "majority": evaluate_majority
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Path to evaluation data in json", required=True)
    parser.add_argument("--model", type=str, default="random", choices=list(MODELS_MAP.keys()), help="Model to use for evaluation")
    parser.add_argument("--output-dir", type=str, help="Output directory for evaluation results", default="outputs")
    parser.add_argument("--seed", type=int, help="Seed for random baseline")

    args = parser.parse_args()

    outputs = MODELS_MAP[args.model](args.datapath, seed=args.seed)

    datapath = pathlib.Path(args.datapath)
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{datapath.stem}_random_{generate_unique_id()}.json")
    print(f"Writing to {output_path}")

    write_json(outputs, output_path)

if __name__ == "__main__":
    main()