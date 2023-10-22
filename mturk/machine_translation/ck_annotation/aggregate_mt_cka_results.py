import sys
import argparse
import pathlib
from tqdm import tqdm

sys.path.append("../../..")

from utils import read_json, write_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", type=str, help="Path to processed json results.", required=True)

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
                "knowledge": [],
            }

            if "sentence" in result:
                agg_result["sentence"] = result["sentence"]
                agg_result["true_referent"] = result["true_referent"]
                agg_result["false_referent"] = result["false_referent"]
            else:
                agg_result["plausible_sentence"] = result["plausible_sentence"]
                agg_result["implausible_sentence"] = result["implausible_sentence"]

            agg_results[result["hit_id"]] = agg_result

        agg_result["assignments"].append(result["assignment_id"])
        agg_result["workers"].append(result["worker_id"])
        agg_result["knowledge"].extend(result["knowledge"])
    
    results_path = pathlib.Path(args.results_path)
    agg_results_path = results_path.parent / f"{results_path.stem}_agg.json"

    write_json(list(agg_results.values()), agg_results_path)

if __name__ == "__main__":
    main()