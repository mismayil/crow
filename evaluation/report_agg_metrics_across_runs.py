import argparse
import sys
import numbers
import pathlib
from collections import defaultdict

sys.path.append("../")

from utils import read_json, write_json, find_json_files, get_avg_metrics_from_dicts

TASKS = ["dialogue", "summarization", "intent", "safety", "stance", "mt_en_fr", "mt_en_de", "mt_en_ru", "mt_zh_en"]

def compute_avg_metrics(model, results_dirs):
    task_metrics_lst_map = defaultdict(list)
    agg_metrics_lst = []

    for results_dir in results_dirs:
        seen_tasks = set()
        files = find_json_files(results_dir)

        for file in files:
            if model in file:
                metrics = read_json(file)
                if "agg_metrics" in file:
                    agg_metrics_lst.append(metrics)
                elif "metrics" in file:
                    task = [task for task in TASKS if task in file][0]

                    if task not in seen_tasks:
                        seen_tasks.add(task)
                        task_metrics_lst_map[task].append(metrics)
                    else:
                        raise ValueError(f"Found metrics for {task} twice")
    
    avg_metrics_per_task = {}

    for task, task_metrics_lst in task_metrics_lst_map.items():
        if task_metrics_lst:
            avg_metrics_per_task[task] = get_avg_metrics_from_dicts(task_metrics_lst)
        
    global_avg_metrics = get_avg_metrics_from_dicts(agg_metrics_lst)

    return global_avg_metrics, avg_metrics_per_task
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model name", required=True)
    parser.add_argument("--results-dirs", nargs="+", type=str, help="Model results directories", required=True)
    parser.add_argument("--output-dir", type=str, help="Output dir for metrics", required=True)

    args = parser.parse_args()

    global_avg_metrics, avg_metrics_per_task = compute_avg_metrics(args.model, args.results_dirs)

    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    global_avg_metrics_path = f"{args.output_dir}/{args.model}_avg_metrics.json"
    write_json(global_avg_metrics, global_avg_metrics_path)

    for task, avg_task_metrics in avg_metrics_per_task.items():
        avg_task_metrics_path = f"{args.output_dir}/{args.model}_{task}_avg_metrics.json"
        write_json(avg_task_metrics, avg_task_metrics_path)

if __name__ == "__main__":
    main()