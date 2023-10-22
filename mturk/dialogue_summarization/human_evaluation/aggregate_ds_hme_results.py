import sys
import argparse
import pathlib
from tqdm import tqdm
import pandas as pd

sys.path.append("../../..")

from utils import read_json, write_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", type=str, help="Path to csv results.", required=True)

    args = parser.parse_args()
    results = read_json(args.results_path)
    agg_results = {}

    for result in tqdm(results, desc="Aggregating results", total=len(results)):
        agg_result = agg_results.get(result["hit_id"])

        if not agg_result:
            agg_result = {
                "data_id": result["data_id"],
                "hit_id": result["hit_id"],
                "assignments": [],
                "workers": [],
                "dialogue": result["dialogue"],
                "summary": result["summary"],
                "summary_id": result["summary_id"],
                "label": result["label"],
                "answers": []
            }
            agg_results[result["hit_id"]] = agg_result
        
        agg_result["assignments"].append(result["assignment_id"])
        agg_result["workers"].append(result["worker_id"])
        agg_result["answers"].append(result["answer"])
        agg_result["agreement"] = (len(set(agg_result["answers"])) == 1)
        agg_result["prediction"] = agg_result["answers"][0]
        agg_result["correct"] = (agg_result["prediction"] == agg_result["label"]) if agg_result["agreement"] else None
    
    disagreements = []

    for result in agg_results.values():
        if not result["agreement"] or len(result["answers"]) < 2:
            disagreements.append({
                "data_id": result["data_id"],
                "hit_id": result["hit_id"],
                "assignments": result["assignments"],
                "workers": result["workers"],
                "dialogue": result["dialogue"],
                "summary": result["summary"],
                "summary_id": result["summary_id"],
                "answers": result["answers"],
                "agreement": result["agreement"],
                "prediction": None
            })

    results_path = pathlib.Path(args.results_path)
    agg_results_path = results_path.parent / f"{results_path.stem}_agg.json"
    disagreements_path = results_path.parent / f"{results_path.stem}_disagreements.json"

    write_json(list(agg_results.values()), agg_results_path)
    write_json(disagreements, disagreements_path)
    pd.DataFrame(disagreements).to_csv(results_path.parent / f"{results_path.stem}_disagreements.csv", index=False)

if __name__ == "__main__":
    main()