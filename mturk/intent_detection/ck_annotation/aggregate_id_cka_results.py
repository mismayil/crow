import sys
import argparse
import pathlib
from tqdm import tqdm

sys.path.append("../../..")

from utils import read_json, write_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", type=str, help="Path to json results.", required=True)

    args = parser.parse_args()
    results = read_json(args.results_path)
    agg_results = {}

    for result in tqdm(results, desc="Aggregating results", total=len(results)):
        for _, kg in enumerate(result["knowledge"]):
            selected_intents = [intent for intent in result["intents"] if intent["id"] == kg["intent_id"]]
            selected_intent_ids = tuple([intent["id"] for intent in selected_intents])
            agg_tuple = (result["hit_id"], selected_intent_ids)
            agg_result = agg_results.get(agg_tuple)

            if not agg_result:
                agg_result = {
                    "data_id": result["data_id"],
                    "hit_id": result["hit_id"],
                    "assignments": [],
                    "workers": [],
                    "headline": result["headline"],
                    "intents": selected_intents,
                    "knowledge": [],
                    "no_ck": 0
                }
                agg_results[agg_tuple] = agg_result
            
            agg_result["assignments"].append(result["assignment_id"])
            agg_result["workers"].append(result["worker_id"])
            agg_result["knowledge"].append(kg)
            agg_result["no_ck"] += int(result.get("no_ck", False))
    
    results_path = pathlib.Path(args.results_path)
    agg_results_path = results_path.parent / f"{results_path.stem}_agg.json"

    write_json(list(agg_results.values()), agg_results_path)

if __name__ == "__main__":
    main()