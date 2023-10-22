import sys
import argparse
import pathlib
from tqdm import tqdm

sys.path.append("../..")

from utils import read_json, write_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-path", type=str, help="Path to report json.", required=True)
    parser.add_argument("--results-path", type=str, help="Path to results json.", required=True)
    parser.add_argument("--metrics", type=str, nargs="+", choices=["accuracy", "f1", "recall", "precision"], help="Metrics for qualification.", default=["precision"])
    parser.add_argument("--precision-th", type=float, help="Precision threshold.", default=0.8)
    parser.add_argument("--recall-th", type=float, help="Recall threshold.", default=0.8)
    parser.add_argument("--f1-th", type=float, help="F1 threshold.", default=0.8)
    parser.add_argument("--accuracy-th", type=float, help="Accuracy threshold.", default=0.8)

    args = parser.parse_args()

    report = read_json(args.report_path)
    results = read_json(args.results_path)

    results_map = {}

    for result in results:
        results_map[result["worker_id"]] = result
    
    qualified_workers = []

    for worker in tqdm(report["workers"], desc="Qualifying workers"):
        qualified = True
        for metric in args.metrics:
            if worker[metric] < getattr(args, f"{metric}_th"):
                qualified = False
                break
        result = results_map[worker["worker_id"]]
        knowledge = result["oq_odd_questions"][0]["knowledge"]
        qualified_workers.append({
            "worker_id": worker["worker_id"],
            "hit_id": worker["hit_id"],
            "assignment_id": worker["assignment_id"], 
            "score": int(worker["f1"]*100),
            "f1": worker["f1"],
            "precision": worker["precision"],
            "recall": worker["recall"],
            "accuracy": worker["accuracy"],
            "qualified": qualified,
            "bonus": len(knowledge)-1,
            "reason": "Bonus for additional knowledge annotations.",
            "feedback": result["feedback"],
            "ee_time": result["worker_ee"],
            "knowledge": knowledge
        })
    
    print("Number of qualified workers: {}".format(len([worker for worker in qualified_workers if worker["qualified"]])))

    report_path = pathlib.Path(args.report_path)
    thresholds = [getattr(args, f"{metric}_th") for metric in args.metrics]
    metrics_str = "_".join([f"{m[0]}{th}" for m, th in zip(args.metrics, thresholds)])
    qualified_path = report_path.parent / f"{report_path.stem.replace('_report', '')}_qfd_{metrics_str}.json"

    write_json(qualified_workers, qualified_path)

if __name__ == "__main__":
    main()