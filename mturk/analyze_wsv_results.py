import sys
import argparse
from tqdm import tqdm
import numpy as np
import pathlib
import krippendorff as krip

sys.path.append("..")

from utils import read_json, write_json, fleiss_kappa

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsv-results-path", type=str, help="Path to WSV json results.", required=True)
    parser.add_argument("--wsg-results-path", type=str, help="Path to WSG json results.", required=True)
    parser.add_argument("--alt-target-attr", type=str, help="Attribute to use for alt target.", default="alt_targets")
    parser.add_argument("--krip-level", type=str, default="nominal", help="Level of measurement (nominal, ordinal, interval, ratio)")
    parser.add_argument("--num-annotators", type=int, default=3, help="Number of annotators per result.")
    
    args = parser.parse_args()

    wsv_results = read_json(args.wsv_results_path)
    wsg_results = read_json(args.wsg_results_path)
    work_times = [res["worker_time"] for res in wsv_results]
    ee_times = [res["worker_ee"] for res in wsv_results]
    unique_workers = set([res["worker_id"] for res in wsv_results])

    report = {
        "num_results": len(wsv_results),
        "num_hits": len(set([res["hit_id"] for res in wsv_results])),
        "num_annotations": sum([len(res[args.alt_target_attr]) for res in wsv_results]),
        "num_new_annotations": sum([len(res[f"new_{args.alt_target_attr}"]) for res in wsv_results]),
        "mean_work_time": sum(work_times) / len(wsv_results),
        "mean_ee_time": sum(ee_times) / len(wsv_results),
        "max_work_time": max(work_times),
        "max_ee_time": max(ee_times),
        "min_work_time": min(work_times),
        "min_ee_time": min(ee_times),
        "num_workers": len(unique_workers),
        "workers": [{"worker_id": worker_id, "work_time": 0, "ee_time": 0, "num_hits": 0, "num_annotations": 0, "num_reviewed_annotations": 0, "num_new_annotations": 0, "points": 0, "quality": 0} for worker_id in unique_workers],
        "annotation_quality": 0
    }

    target_map = {}
    worker_map = {worker["worker_id"]: worker for worker in report["workers"]}

    for wsg_result in wsg_results:
        for target in wsg_result[args.alt_target_attr]:
            target_map[target["id"]] = {**target, "num_votes": 0, "votes": []}

    for r_index, result in tqdm(enumerate(wsv_results), total=len(wsv_results), desc="Analyzing results"):
        worker = worker_map[result["worker_id"]]
        worker["num_hits"] += 1
        worker["num_annotations"] += len(result[args.alt_target_attr])
        worker["num_new_annotations"] += len(result[f"new_{args.alt_target_attr}"])
        worker["work_time"] += result["worker_time"]
        worker["ee_time"] += result["worker_ee"]

        for target in result[args.alt_target_attr]:
            wsg_target = target_map.get(target["id"])
            selected = target.get("selected", False)
            wsg_target["votes"].append(int(selected))

            if wsg_target:
                quality = wsg_target.get("quality")

                if quality is not None:
                    worker["num_reviewed_annotations"] += 1
                    points = quality * (1 if selected else -1)
                    worker["points"] += points

                    if worker.get(f"num_points_{points}") is None:
                        worker[f"num_points_{points}"] = 0
                    
                    worker[f"num_points_{points}"] += 1
                    wsg_target["num_votes"] += int(selected)

    for worker in report["workers"]:
        worker["work_time"] = round(worker["work_time"] / worker["num_hits"], 2)
        worker["ee_time"] = round(worker["ee_time"] / worker["num_hits"], 2)
        if worker["num_reviewed_annotations"] > 0:
            worker["quality"] = round(worker["points"] / worker["num_reviewed_annotations"], 2)

    report["workers"] = sorted(report["workers"], key=lambda x: x["quality"], reverse=True)
    total_points = sum([worker["points"] for worker in report["workers"]])
    total_reviewed_annotations = sum([worker["num_reviewed_annotations"] for worker in report["workers"]])

    if total_reviewed_annotations > 0:
        report["annotation_quality"] = round(total_points / total_reviewed_annotations, 2)
        report["annotation_accuracy"] = round(sum([1 if (target["num_votes"] >= 2 and target["quality"] > 0) else 0 for target in target_map.values() if "quality" in target]) / total_reviewed_annotations, 2)

    vote_matrix = np.array([target["votes"] for target in target_map.values() if len(target["votes"]) == args.num_annotators])
    report["krippendorff_alpha"] = round(krip.alpha(vote_matrix.T, level_of_measurement=args.krip_level), 2)
    report["fleiss_kappa"] = round(fleiss_kappa(vote_matrix), 2)

    results_path = pathlib.Path(args.wsv_results_path)
    report_path = results_path.parent / f"{results_path.stem}_report.json"

    write_json(report, report_path)

if __name__ == "__main__":
    main()