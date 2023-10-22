import sys
import argparse
from tqdm import tqdm
import pathlib
import pandas as pd

sys.path.append("..")

from utils import read_json, write_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", type=str, help="Path to aggregated HME json results.", required=True)
    parser.add_argument("--resolution-path", type=str, help="Path to resolved HME csv results.")
    parser.add_argument("--num-annotators", type=int, help="Number of annotators per data point.", default=2)
    
    args = parser.parse_args()

    results = read_json(args.results_path)
    resolutions = pd.read_csv(args.resolution_path)

    resolutions = resolutions.to_dict("records")

    resolution_map = {}

    if resolutions:
        for resolution in resolutions:
            resolution_map[resolution["hit_id"]] = resolution
    
    for result in tqdm(results, total=len(results), desc="Finalizing results"):
        final_answer = result["answers"][0]
        if result["agreement"] and len(result["answers"]) == args.num_annotators:
            final_answer = result["answers"][0]
        else:
            if resolutions:
                resolution = resolution_map[result["hit_id"]]
                final_answer = resolution["prediction"]

        result["prediction"] = final_answer
        result["correct"] = (result["prediction"] == result["label"])

    results_path = pathlib.Path(args.results_path)
    final_results_path = results_path.parent / f"{results_path.stem}_resolved.json"

    write_json(results, final_results_path)

if __name__ == "__main__":
    main()