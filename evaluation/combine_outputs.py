import argparse
import sys
from collections import defaultdict, Counter
import pathlib
from tqdm import tqdm


sys.path.append("../")

from utils import read_json, write_json, find_json_files

TASKS = ["dialogue", "summarization", "intent", "safety", "stance", "mt_en_fr", "mt_en_de", "mt_en_ru", "mt_zh_en"]

def combine_outputs(model, results_dirs):
    task_output_lst_map = {task: [] for task in TASKS}

    for results_dir in tqdm(results_dirs, desc="Collecting outputs"):
        files = find_json_files(results_dir)
        seen_tasks = set()

        for file in files:
            if model in file:
                if "metrics" not in file:
                    outputs = read_json(file)
                    task = [task for task in TASKS if task in file][0]

                    if task not in seen_tasks:
                        seen_tasks.add(task)
                        task_output_lst_map[task].append(outputs)
                    else:
                        raise ValueError(f"Found metrics for {task} twice")

    combined_outputs_per_task = {task: [] for task in TASKS}

    for task, output_lst in tqdm(task_output_lst_map.items(), desc="Combining outputs"):
        output_map = defaultdict(list)

        for outputs in output_lst:
            for result in outputs["data"]:
                output_map[result["instance_id"]].append(result)
        
        for _, output_lst in output_map.items():
            answer = output_lst[0]["answer"]
            responses = [output["response"] for output in output_lst]
            response = Counter(responses).most_common(1)[0][0]
            combined_outputs_per_task[task].append({
                **output_lst[0],
                "response": response,
                "correct": response == answer
            })
        
    return combined_outputs_per_task

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model name", required=True)
    parser.add_argument("--results-dirs", type=str, nargs="+", help="Model results directories", required=True)
    parser.add_argument("--output-dir", type=str, help="Output dir for combined results", required=True)

    args = parser.parse_args()

    combined_outputs_per_task = combine_outputs(args.model, args.results_dirs)

    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for task, combined_outputs in combined_outputs_per_task.items():
        outputs_path = f"{args.output_dir}/{args.model}_{task}_combined_outputs.json"
        write_json({"data": combined_outputs}, outputs_path)

if __name__ == "__main__":
    main()