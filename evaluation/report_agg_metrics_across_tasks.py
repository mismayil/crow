import argparse
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pprint
from collections import defaultdict

sys.path.append("../")

from utils import read_json, write_json, find_json_files

TASKS = ["dialogue", "summarization", "intent", "safety", "stance", "mt_en_fr", "mt_en_de", "mt_en_ru", "mt_zh_en"]
MT_TASKS = ["mt_en_fr", "mt_en_de", "mt_en_ru", "mt_zh_en"]

def compute_agg_metrics(model, results_dir):
    agg_metrics = defaultdict(list)
    agg_dim_metrics = defaultdict(list)
    agg_usage_metrics = defaultdict(list)
    agg_cost_metrics = defaultdict(list)
    non_mt_agg_metrics = defaultdict(list)
    non_mt_agg_dim_metrics = defaultdict(list)

    files = find_json_files(results_dir)

    seen_tasks = set()

    for file in files:
        if model in file:
            metrics = read_json(file)

            if "data" not in metrics and "agg_metrics" not in file:
                task = [task for task in TASKS if task in file][0]

                if task not in seen_tasks:
                    seen_tasks.add(task)
                    for metric, value in metrics.items():
                        if metric == "dimensions":
                            for dim, dim_value in value.items():
                                agg_dim_metrics[dim].append(dim_value)
                        elif metric == "usage":
                            for usage_metric, usage_value in value.items():
                                agg_usage_metrics[usage_metric].append(usage_value)
                        elif metric == "cost":
                            for cost_metric, cost_value in value.items():
                                agg_cost_metrics[cost_metric].append(cost_value)
                        else:
                            agg_metrics[metric].append(value)
                        
                        if task not in MT_TASKS:
                            if metric == "dimensions":
                                for dim, dim_value in value.items():
                                    non_mt_agg_dim_metrics[dim].append(dim_value)
                            elif metric != "usage" and metric != "cost":
                                non_mt_agg_metrics[metric].append(value)
                else:
                    raise ValueError(f"Found metrics for {task} twice")

    metrics = {"dimensions": {}, "non_mt_dimensions": {}, "cost": {}, "usage": {}}

    for metric, values in agg_metrics.items():
        metrics[metric] = sum(values) / len(values)
    
    for dim, values in agg_dim_metrics.items():
        metrics["dimensions"][dim] = sum(values) / len(values)

    for usage_metric, values in agg_usage_metrics.items():
        metrics["usage"][usage_metric] = sum(values)

    for cost_metric, values in agg_cost_metrics.items():
        metrics["cost"][cost_metric] = sum(values)

    for metric, values in non_mt_agg_metrics.items():
        metrics[f"non_mt_{metric}"] = sum(values) / len(values)
    
    for dim, values in non_mt_agg_dim_metrics.items():
        metrics["non_mt_dimensions"][dim] = sum(values) / len(values)

    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model name", required=True)
    parser.add_argument("--results-dir", type=str, help="Model results", required=True)

    args = parser.parse_args()

    metrics = compute_agg_metrics(args.model, args.results_dir)

    metrics_path = f"{args.results_dir}/{args.model}_agg_metrics.json"
    write_json(metrics, metrics_path)

    pprint.pprint(metrics)

if __name__ == "__main__":
    main()