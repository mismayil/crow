import sys
import argparse
from tqdm import tqdm
import numpy as np
import pathlib
from collections import defaultdict

sys.path.append("..")

from utils import read_json, write_json, text_to_wordset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", type=str, help="Path to json results.", required=True)
    parser.add_argument("--alt-target-attr", type=str, help="Attribute to use for alt target.", default="alt_targets")

    args = parser.parse_args()

    results = read_json(args.results_path)
    work_times = [res["worker_time"] for res in results]
    ee_times = [res["worker_ee"] for res in results]
    unique_workers = list(set([res["worker_id"] for res in results]))

    report = {
        "num_results": len(results),
        "num_hits": len(set([res["hit_id"] for res in results])),
        "num_annotations": sum([len(res[args.alt_target_attr]) for res in results]),
        "mean_work_time": sum(work_times) / len(results),
        "mean_ee_time": sum(ee_times) / len(results),
        "max_work_time": max(work_times),
        "max_ee_time": max(ee_times),
        "min_work_time": min(work_times),
        "min_ee_time": min(ee_times),
        "num_workers": len(unique_workers),
        "workers": [{"worker_id": worker_id, "work_time": 0, "ee_time": 0, "num_hits": 0, "num_annotations": 0, "num_reviewed_annotations": 0, "points": 0, "quality": 0} for worker_id in unique_workers],
    }

    worker_map = {worker["worker_id"]: worker for worker in report["workers"]}
    target_map = defaultdict(list)

    for _, result in tqdm(enumerate(results), total=len(results), desc="Analyzing results"):
        worker = worker_map[result["worker_id"]]
        worker["num_hits"] += 1
        worker["num_annotations"] += len(result[args.alt_target_attr])
        worker["work_time"] += result["worker_time"]
        worker["ee_time"] += result["worker_ee"]

        for target in result[args.alt_target_attr]:
            wordset = text_to_wordset(target["text"])
            target_map[tuple(wordset)].append({"worker_id": result["worker_id"], args.alt_target_attr: target["text"]})
            
            quality = target.get("quality")

            if quality is not None:
                worker["num_reviewed_annotations"] += 1
                worker["points"] += quality

                if worker.get(f"num_points_{quality}") is None:
                    worker[f"num_points_{quality}"] = 0
                
                worker[f"num_points_{quality}"] += 1

    for worker in report["workers"]:
        worker["work_time"] = round(worker["work_time"] / worker["num_hits"], 2)
        worker["ee_time"] = round(worker["ee_time"] / worker["num_hits"], 2)
        if worker["num_reviewed_annotations"] > 0:
            worker["quality"] = round(worker["points"] / worker["num_reviewed_annotations"], 2)

    report["workers"] = sorted(report["workers"], key=lambda x: x["quality"], reverse=True)
    report["num_reviewed_annotations"] = sum([worker["num_reviewed_annotations"] for worker in report["workers"]])
    report["quality_distribution"] = {f"num_points_{i}": sum([worker.get(f"num_points_{i}", 0) for worker in report["workers"]]) for i in range(-2, 3)}
    report["target_map"] = [res for _, res in target_map.items() if len(res) > 1]
    
    results_path = pathlib.Path(args.results_path)
    report_path = results_path.parent / f"{results_path.stem}_report.json"

    write_json(report, report_path)

if __name__ == "__main__":
    main()