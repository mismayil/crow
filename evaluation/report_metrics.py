import argparse
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pprint
from collections import defaultdict
import re
from tqdm import tqdm
import pathlib

sys.path.append("../")

from utils import read_json, write_json, CK_DIMENSIONS, find_json_files, MODEL_COSTS, MODEL_ENCODINGS, num_tokens_from_string

TASK_TARGET_MAP = {
    "dialogue": "final_turn", 
    "summarization": "summary", 
    "stance": "instance" , 
    "intent": "intent", 
    "safety": "action", 
    "translation": "translation",
    "mt_en_fr": "translation", 
    "mt_en_de": "translation", 
    "mt_en_ru": "translation", 
    "mt_zh_en": "translation",
}

TASKS = ["dialogue", "summarization", "intent", "safety", "stance", "mt_en_fr", "mt_en_de", "mt_en_ru", "mt_zh_en"]

def get_prediction(response, template):
    if template in ["bcq", "bcq_with_kg"]:
        pred = 1 if response.strip().lower() == "yes" else 0
    elif template in ["bcq_cot", "bcq_cot_with_kg"]:
        match = re.search("<Answer>(?P<pred>.*)</Answer>", response)
        pred = 0

        if match:
            pred = match["pred"].strip().lower()
            pred = 1 if pred == "yes" else 0
    else:
        raise ValueError(f"Template {template} not supported for evaluation.")

    return pred

def compute_usage(sample, model):
    if model not in MODEL_COSTS:
        return None, None

    usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }

    if "usage" in sample:
        usage = sample["usage"]
    else:
        prompt_tokens = num_tokens_from_string(sample["prompt"], model)
        completion_tokens = num_tokens_from_string(sample["response"], model)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    
    input_cost = usage["prompt_tokens"] * MODEL_COSTS[model]["input"]
    output_cost = usage["completion_tokens"] * MODEL_COSTS[model]["output"]

    return usage, {
        "input": input_cost,
        "output": output_cost,
        "total": input_cost + output_cost
    }

def compute_metrics(results, task):
    metrics = {}
    predictions = []
    references = []

    usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }

    cost = {
        "input": 0,
        "output": 0,
        "total": 0
    }
    
    pred_ref_map = {}
    dimension_pred_ref_map = {dim: {"predictions": [], "references": []} for dim in CK_DIMENSIONS}
    data_id_attr = "subdata_id" if task == "dialogue" else "data_id"

    for result in results["data"]:
        response_attr = "response" if "response" in result else "generated_output"

        if response_attr in result:
            ref = 1 if result["answer"].lower() == "yes" else 0
            pred = get_prediction(result[response_attr], result["type"])
            references.append(ref)
            predictions.append(pred)
            pred_ref_item = pred_ref_map.get(result[data_id_attr], True)

            pred_ref_map[result[data_id_attr]] = pred_ref_item & (pred == ref)

            target = result[TASK_TARGET_MAP[task]]

            if "knowledge" in target:
                for kg in target["knowledge"]:
                    dimension = kg["dimension"]
                    dimension_pred_ref_map[dimension]["predictions"].append(pred)
                    dimension_pred_ref_map[dimension]["references"].append(ref)
            
            sample_usage, sample_cost = compute_usage(result, results["metadata"]["model"])

            if sample_usage:
                usage["prompt_tokens"] += sample_usage["prompt_tokens"]
                usage["completion_tokens"] += sample_usage["completion_tokens"]
                usage["total_tokens"] += sample_usage["total_tokens"]

            if sample_cost:
                cost["input"] += sample_cost["input"]
                cost["output"] += sample_cost["output"]
                cost["total"] += sample_cost["total"]
    
    metrics["exact_match"] = sum(pred_ref_map.values()) / len(pred_ref_map)
    metrics["macro_f1"] = f1_score(references, predictions, average='macro')
    metrics["accuracy"] = accuracy_score(references, predictions)
    metrics["precision"] = precision_score(references, predictions)
    metrics["recall"] = recall_score(references, predictions)
    metrics["dimensions"] = {}

    for dim, pred_ref in dimension_pred_ref_map.items():
        if pred_ref["references"]:
            metrics["dimensions"][dim] = f1_score(pred_ref["references"], pred_ref["predictions"], average="macro")

    metrics["usage"] = usage
    metrics["cost"] = cost

    return metrics

def report_metrics(results_files):
    for results_file in tqdm(results_files, total=len(results_files)):
        results = read_json(results_file)
        
        try:
            if "data" in results:
                task = [task for task in TASKS if task in results_file][0]
                metrics = compute_metrics(results, task)
                write_json(metrics, results_file.replace(".json", "_metrics.json"))
        except Exception as e:
            print(results_file)
            raise e

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", type=str, help="Path to evaluation results file in json or directory", required=True)

    args = parser.parse_args()

    files_to_process = []

    results_path = pathlib.Path(args.results_path)

    if results_path.is_file():
        files_to_process.append(args.results_path)
    else:
        files_to_process.extend(find_json_files(args.results_path))

    report_metrics(files_to_process)

if __name__ == "__main__":
    main()