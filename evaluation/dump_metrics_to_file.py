import argparse
import sys
import pathlib
import csv
from tqdm import tqdm
from collections import defaultdict

sys.path.append("../")

from utils import find_json_files, read_json
from evaluation.config import MODELS, TASKS

MODEL_LATEX_MAP = {
    "majority": "Majority",
    "random": "Random",
    "llama7": "LLaMa-7B",
    "llama13": "LLaMa-13B",
    "llama33": "LLaMa-33B",
    "flan-t5": "Flan-T5-11B",
    "alpaca": "Alpaca",
    "flan-alpaca": "Flan-Alpaca",
    "vicuna": "Vicuna",
    "stable-vicuna": "Stable-Vicuna",
    "mt0": "mT0",
    "bloomz7": "BloomZ-7B",
    "palm": "PaLM 1",
    "text-davinci-003": "GPT-3",
    "gpt-4": "GPT-4"
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs="+", default=MODELS, choices=MODELS, help="Models to get results from.")
    parser.add_argument("--runs", type=str, nargs="+", required=True, help="Runs to get results from. Must be in the format of <model>:<run>. If <model> is not specified, it will be assumed to be default run for all other models.")
    parser.add_argument("--output-path", type=str, required=True, help="Output path to write results to.")
    parser.add_argument("--output-format", type=str, choices=["csv", "latex"], default="csv", help="Format to write results in.")
    parser.add_argument("--metrics", type=str, nargs="+", default=["macro_f1", "exact_match"], help="Metrics to write to file.")
    parser.add_argument("--base-outputs-dir", type=str, default="outputs", help="Base directory to look for runs in.")

    args = parser.parse_args()

    run_map = defaultdict(list)
    metrics = []

    for run_str in args.runs:
        run_split = run_str.split(":")
        if len(run_split) == 1:
            run_map["default"].append(run_split[0])
        elif len(run_split) == 2:
            run_key = "default" if run_split[0].strip() == "" else run_split[0]
            run_map[run_key].append(run_split[1])
        else:
            raise ValueError("Invalid run string: {}".format(run_str))

    for model in tqdm(args.models, total=len(args.models), desc="Collecting metrics"):
        runs = run_map.get(model, run_map["default"])

        for run in runs:
            run_dir = pathlib.Path(f"{args.base_outputs_dir}/{model}/{run}") 
            
            if not run_dir.exists():
                print(f"Run directory {run_dir} does not exist. Skipping run {run} for model {model}.")
                continue

            run_files = find_json_files(run_dir)
            
            model_metrics = {"model": model, "run": run, **{task: "--" for task in TASKS}}

            for run_file in run_files:
                run_path = pathlib.Path(run_file)
                
                if run_path.name == f"{model}_agg_metrics.json" or run_path.name == f"{model}_avg_metrics.json":
                    run_metrics = read_json(run_file)
                    non_mt_metrics = []
                    with_mt_metrics = []
                    
                    for metric in args.metrics:
                        non_mt_metrics.append(run_metrics[f"non_mt_{metric}"])
                        with_mt_metrics.append(run_metrics[metric])

                    model_metrics["non_mt_crow_score"] = " / ".join([f"{m*100:.1f}" for m in non_mt_metrics])
                    model_metrics["crow_score"] = " / ".join([f"{m*100:.1f}" for m in with_mt_metrics])
                elif "metrics" in run_file:
                    task = [t for t in TASKS if t in run_file][0]
                    run_metrics = read_json(run_file)
                    task_metrics = []
                    
                    for metric in args.metrics:
                        if metric not in run_metrics:
                            raise ValueError(f"Metric {metric} not found in {run_file}.")

                        task_metrics.append(run_metrics[metric])
                    
                    model_metrics[task] = " / ".join([f"{m*100:.1f}" for m in task_metrics])
        
            metrics.append(model_metrics)
    
    pathlib.Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    if args.output_format == "csv":
        with open(args.output_path, "w") as f:
            writer = csv.DictWriter(f, fieldnames=["model"] + ["run"] + TASKS + ["non_mt_crow_score", "crow_score"])
            writer.writeheader()
            writer.writerows(metrics)
    elif args.output_format == "latex":
        with open(args.output_path, "w") as f:
            for metric in metrics:
                model_name = MODEL_LATEX_MAP.get(metric["model"], metric["model"])
                run_name = metric["run"]
                
                if "cot" in run_name:
                    model_name = f"{model_name}-CoT"

                non_mt_crow_score = metric["non_mt_crow_score"]
                crow_score = metric["crow_score"]
                task_scores = [metric[task] for task in TASKS]
                f.write(f"\\textbf{{{model_name}}} & {' & '.join(task_scores)} & {non_mt_crow_score} & {crow_score} \\\\\n")

                if metric["model"] in ["random", "stable-vicuna", "bloomz7"]:
                    f.write("\\midrule\n")

if __name__ == "__main__":
    main()