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
        for _, target in enumerate(result["alt_targets"]):
            selected_knowledge = [kg for kg in result["knowledge"] if kg["id"] in target["knowledge_ids"]]   
            selected_kg_ids = tuple([kg["id"] for kg in selected_knowledge])
            agg_tuple = (result["hit_id"], selected_kg_ids)
            agg_result = agg_results.get(agg_tuple)

            if not agg_result:
                agg_result = {
                    "data_id": result["data_id"],
                    "hit_id": result["hit_id"],
                    "assignments": [],
                    "workers": [],
                    "dialogue": result["dialogue"],
                    "final_turn": result["final_turn"],
                    "final_turn_prefix": result["final_turn_prefix"],
                    "knowledge": selected_knowledge,
                    "alt_targets": [],
                    "no_alt_target": 0
                }
                agg_results[agg_tuple] = agg_result
            
            agg_result["assignments"].append(result["assignment_id"])
            agg_result["workers"].append(result["worker_id"])
            agg_result["alt_targets"].append(target)
            agg_result["no_alt_target"] += int(result.get("no_alt_target", False))
    
    results_path = pathlib.Path(args.results_path)
    agg_results_path = results_path.parent / f"{results_path.stem}_agg.json"

    write_json(list(agg_results.values()), agg_results_path)

if __name__ == "__main__":
    main()